"""unfold.py

Tools to construct an "unfolded" simulation cell for a periodic system,
which simply means all posititions in the bulk within a shell of thickness
given by a `cutoff`, which is needed for heat flux calculations.

We split this task into two: the calculation of the "recipe" for the unfolding,
and the execution of that recipe. We do this in order to cache the former, and
in order to make sure that the latter remains differentiable in the expected way.

The "recipe" for unfolding, the `Unfolding` consists of an index array indicating
which original atom is replicated, `replica_offsets` defining which combination of
lattice vectors is applied for that replica, `wrap_offsets` that tell us which
combination of lattice vectors can be applied to return each position in `positions`
into the simulation cell (this will mostly be 0, but we want to treat cases where atoms
have strayed out of the cell), and a mask to mask out positions that are due to padding.

In order to be able to cache things, similar to `neighborlist`, we add a `skin` that
gives us some additional cutoff to let atoms move before recomputing things.

The first calculation of an `Unfolding` is not jittable since it returns an unknown number
of replicas. Subsequent application of that unfolding is jittable. Similar to the neighborlist,
we allow some additional (fake) positions -- for this reason, computing the unfolding is also
jittable, provided the number of replicated positions is not too large.

"""


from jax import jit, vmap
import jax.numpy as jnp
from jax.lax import stop_gradient, cond

from collections import namedtuple
from functools import partial

from glp import comms
from glp.periodic import project_on_normals, get_heights, to_frac
from glp.utils import cast


Unfolding = namedtuple(
    "Unfolding",
    (
        "replica_idx",  # [M] ; original atom of which this is a replica
        "replica_offsets",  # [M, 3] ; which cell offsets to apply
        "wrap_offsets",  # [N, 3] ; offsets to apply to return positions to cell
        "padding_mask",  # [M] ; if False this is a fake position
        "reference_positions",  # [N, 3] ; positions at last update
        "reference_cell",  # [3, 3] ; cell at last update
        "overflow",  # if True, unfolding is invalid
        "updated",  # if True, this unfolding has just been updated
    ),
)

# apply unfolding


def unfold(positions, cell, unfolding):
    wrapped = wrap(positions, cell, unfolding.wrap_offsets)
    unfolded = replicate(
        wrapped[unfolding.replica_idx], cell, unfolding.replica_offsets
    )

    # avoid spurious gradients to positions[-1]
    unfolded = unfolded * unfolding.padding_mask[:, None]

    return wrapped, unfolded


def replicate(positions, cell, offsets):
    return positions + jnp.einsum("aA,iA->ia", cell, offsets)


def wrap(positions, cell, offsets):
    return positions + jnp.einsum("aA,iA->ia", cell, offsets)


# obtain unfolding


def unfolder(system, cutoff, skin, capacity_multiplier=1.1, debug=False):
    # generating the unfolding is not part of the potential forward computation,
    # but applying it is -- gradients will flow through unfold() but not the rest

    assert capacity_multiplier >= 1.0

    cutoff = cast(stop_gradient(cutoff))
    skin = cast(stop_gradient(skin))

    if system.cell is not None:

        def cell_too_small(cell):
            min_height = jnp.min(get_heights(cell))
            return (cutoff + skin) > min_height

    else:

        def cell_too_small(cell):
            return False

    if cell_too_small(system.cell):
        min_height = jnp.min(get_heights(system.cell))
        comms.warn(
            f"warning: unfolding is only possible up to {min_height:.1f}Å but total cutoff is {cutoff:.1f}+{skin:.1f}={skin+cutoff:.1f} Å"
        )
        comms.warn("this will yield incorrect results!")

    def allocate_fn(system):
        system = stop_gradient(system)

        wrap_offsets = get_wrap_offsets(system.R, system.cell)
        wrapped_positions = wrap(system.R, system.cell, wrap_offsets)

        replicas = get_all_replicas(wrapped_positions, system.cell, cutoff + skin)

        count = jnp.sum(replicas)
        size = int(count * capacity_multiplier) + 1

        replica_idx, replica_offsets, padding_mask, overflow = get_unfolding(
            replicas, size
        )

        return Unfolding(
            replica_idx,
            replica_offsets,
            wrap_offsets,
            padding_mask,
            system.R,
            system.cell,
            overflow,
            jnp.array(False),
        )

    def need_update_fn(system, unfolding):
        movements = system.R - unfolding.reference_positions
        movements = jnp.abs(project_on_normals(system.cell, movements))

        # we just give up if the cell changes -- this should only happen during testing,
        # as the unfolded stuff should only be relevant for NVE
        cell_changes = jnp.abs(system.cell - unfolding.reference_cell).sum()

        return (
            jnp.any(movements >= cast(0.5) * skin)
            | (cell_changes > 0)
            | cell_too_small(system.cell)
        )

    def update_fn(system, unfolding, force_update=False):
        def actual_update_fn(system, unfolding):
            system = stop_gradient(system)

            wrap_offsets = get_wrap_offsets(system.R, system.cell)
            wrapped_positions = wrap(system.R, system.cell, wrap_offsets)

            replicas = get_all_replicas(wrapped_positions, system.cell, cutoff + skin)
            replica_idx, replica_offsets, padding_mask, overflow = get_unfolding(
                replicas, unfolding.replica_idx.shape[0]
            )

            overflow = overflow | cell_too_small(system.cell)

            return Unfolding(
                replica_idx,
                replica_offsets,
                wrap_offsets,
                padding_mask,
                system.R,
                system.cell,
                overflow,
                jnp.array(True),
            )

        def fake_update_fn(system, unfolding):
            return Unfolding(
                unfolding.replica_idx,
                unfolding.replica_offsets,
                unfolding.wrap_offsets,
                unfolding.padding_mask,
                system.R,
                system.cell,
                unfolding.overflow,
                jnp.array(False),
            )

        return cond(
            force_update | need_update_fn(system, unfolding),
            actual_update_fn,
            fake_update_fn,
            system,
            unfolding,
        )

    unfolding = allocate_fn(system)

    if not debug:
        return unfolding, update_fn
    else:
        return unfolding, update_fn, need_update_fn


def get_wrap_offsets(positions, cell):
    # compute the offsets needed to retvrn positions into cell,
    # easily computed by taking the div wrt 1.0 in fractional coords

    frac = to_frac(cell, positions)
    offsets = cast(-1.0) * (frac // cast(1.0)).astype(jnp.int32)

    return offsets


@partial(jit, static_argnums=1)
def get_unfolding(replicas, size):
    padded_replicas = jnp.argwhere(replicas, size=size, fill_value=cast(-1))

    replica_idx = padded_replicas[:, 0]
    replica_offsets = jnp.array([0, 1, -1], dtype=int)[padded_replicas[:, 1:]]

    padding_mask = replica_idx != cast(-1)

    total = jnp.sum(replicas)
    overflow = padded_replicas.shape[0] < total

    return replica_idx, replica_offsets, padding_mask, overflow


def get_all_replicas(positions, cell, cutoff):
    heights = get_heights(cell)

    # [N, 3] ; positions projected onto normals
    projections = project_on_normals(cell, positions)

    # [N, 6] ; is within cutoff of left/right boundary
    collisions = vmap(lambda X: collision(X, heights, cutoff))(projections)

    # [N, 3, 3, 3] ; for each position, which of the 27 possible replicas is needed
    replicas = vmap(collision_to_replica)(collisions)

    return replicas


def collision(X, heights, cutoff):
    x_lo = X[0] <= cutoff
    x_hi = X[0] >= heights[0] - cutoff

    y_lo = X[1] <= cutoff
    y_hi = X[1] >= heights[1] - cutoff

    z_lo = X[2] <= cutoff
    z_hi = X[2] >= heights[2] - cutoff

    return jnp.array([x_lo, x_hi, y_lo, y_hi, z_lo, z_hi], dtype=bool)


def collision_to_replica(collision):
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = collision

    out = jnp.zeros((3, 3, 3), dtype=bool)

    # 6 faces

    out = out.at[+1, 0, 0].set(x_lo)
    out = out.at[-1, 0, 0].set(x_hi)

    out = out.at[0, +1, 0].set(y_lo)
    out = out.at[0, -1, 0].set(y_hi)

    out = out.at[0, 0, +1].set(z_lo)
    out = out.at[0, 0, -1].set(z_hi)

    # 12 edges

    out = out.at[+1, +1, 0].set(x_lo & y_lo)
    out = out.at[+1, -1, 0].set(x_lo & y_hi)

    out = out.at[-1, +1, 0].set(x_hi & y_lo)
    out = out.at[-1, -1, 0].set(x_hi & y_hi)

    out = out.at[0, +1, +1].set(y_lo & z_lo)
    out = out.at[0, +1, -1].set(y_lo & z_hi)

    out = out.at[0, -1, +1].set(y_hi & z_lo)
    out = out.at[0, -1, -1].set(y_hi & z_hi)

    out = out.at[+1, 0, +1].set(x_lo & z_lo)
    out = out.at[+1, 0, -1].set(x_lo & z_hi)

    out = out.at[-1, 0, +1].set(x_hi & z_lo)
    out = out.at[-1, 0, -1].set(x_hi & z_hi)

    # 8 corners

    out = out.at[+1, +1, +1].set(x_lo & y_lo & z_lo)
    out = out.at[+1, +1, -1].set(x_lo & y_lo & z_hi)

    out = out.at[-1, +1, -1].set(x_hi & y_lo & z_hi)
    out = out.at[-1, +1, +1].set(x_hi & y_lo & z_lo)

    out = out.at[+1, -1, +1].set(x_lo & y_hi & z_lo)
    out = out.at[-1, -1, +1].set(x_hi & y_hi & z_lo)

    out = out.at[-1, -1, -1].set(x_hi & y_hi & z_hi)
    out = out.at[+1, -1, -1].set(x_lo & y_hi & z_hi)

    return out
