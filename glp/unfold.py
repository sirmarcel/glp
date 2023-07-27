"""unfold.py

Tools to construct an "unfolded" simulation cell for a periodic system,
which simply means all posititions in the bulk within a shell of thickness
given by a `cutoff`, which is needed for heat flux calculations.

We split this task into two: the calculation of the "recipe" for the unfolding,
and the execution of that recipe. We do this in order to cache the former, and
in order to make sure that the latter remains differentiable in the expected way.

The "recipe" for unfolding, the `Unfolding` consists of an index array indicating
which original atom is replicated, `replica_offsets` defining which combination of
lattice vectors is applied for that replica, and `wrap_offsets` that tell us which
combination of lattice vectors can be applied to return each position in `positions`
into the simulation cell (this will mostly be 0, but we want to treat cases where atoms
have strayed out of the cell).

In order to be able to cache things, similar to `neighborlist`, we add a `skin` that
gives us some additional cutoff to let atoms move before recomputing things.

The first calculation of an `Unfolding` is not jittable since it returns an unknown number
of replicas. Subsequent application of that unfolding is jittable. Checking if we need to
update the unfolding is also jittable.

"""


from collections import namedtuple
from jax import jit, vmap
import jax.numpy as jnp
from jax.lax import stop_gradient

from glp import periodic
from .utils import cast
from .unfold_numpy import _np_unfolding

Unfolding = namedtuple(
    "Unfolding",
    (
        "wrap_offsets",
        "replica_idx",
        "replica_offsets",
        "reference_positions",
        "reference_cell",
    ),
)
Unfolder = namedtuple("Unfolder", ("unfolding", "needs_update"))

# offsets are given as 3-tuples that can be -1, 0, 1, indexing the 3 lattice vectors,
# i.e. [1, 1, 1] means we take an offset of basis_0 + basis_1 + basis_2, etc.


def unfolder(system, cutoff, skin):
    # generating the unfolding is not part of the potential forward computation,
    # but applying it is -- gradients will flow through unfold() but not the rest
    positions = stop_gradient(system.R)
    cell = stop_gradient(system.cell)

    cutoff = cast(stop_gradient(cutoff))
    skin = cast(stop_gradient(skin))

    unfolding = get_unfolding(positions, cell, cutoff + skin)

    @jit
    def needs_update(system, unfolding):
        # consider: filter down to just those within spitting range
        # of the cutoff -- we don't care about movement in the interior
        # (this is probably way too much optimisation)
        movements = system.R - unfolding.reference_positions
        movements = jnp.abs(periodic.project_on_normals(system.cell, movements))

        # we just give up if the cell changes -- this should only happen during testing,
        # as the unfolded stuff should only be relevant for NVE
        cell_changes = jnp.abs(system.cell - unfolding.reference_cell).sum()

        return jnp.any(movements >= cast(0.5) * skin) | (cell_changes > 0)

    return unfolding, needs_update


# stuff to execute a given unfolding


@jit
def unfold(positions, cell, unfolding):
    wrapped = wrap(positions, cell, unfolding.wrap_offsets)
    unfolded = replicate(
        wrapped[unfolding.replica_idx], cell, unfolding.replica_offsets
    )

    return wrapped, unfolded


def wrap(positions, cell, offsets):
    return positions + jnp.einsum("aA,iA->ia", cell, offsets)


def replicate(positions, cell, offsets):
    return positions + jnp.einsum("aA,iA->ia", cell, offsets)


# stuff to compute a new unfolding


def get_unfolding(positions, cell, cutoff):
    """Generate unfolding.

    We do it naively: We generate all 26 replicas and then trim
    down to the ones that are within the cutoff (+skin). This is
    very simple, but somewhat inefficient. The advantage is that
    everything but the trimming down can be jit.
    """

    N = positions.shape[0]

    wrap_offsets = get_wrapping(positions, cell)
    wrapped_positions = wrap(positions, cell, wrap_offsets)

    replica_idx, replica_offsets = _np_unfolding(wrapped_positions, cell, cutoff)

    return Unfolding(
        wrap_offsets,
        jnp.array(replica_idx, dtype=jnp.int32),
        jnp.array(replica_offsets, dtype=jnp.int32),
        positions,
        cell,
    )


## the jittable bits of unfolding generation


@jit
def get_wrapping(positions, cell):
    # compute the offsets needed to retvrn positions into cell,
    # easily computed by taking the div wrt 1.0 in fractional coords

    frac = periodic.to_frac(cell, positions)
    offsets = cast(-1.0) * (frac // cast(1.0)).astype(jnp.int32)

    return offsets
