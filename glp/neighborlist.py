"""neighborlist.py

The role of a neighborlist implementation is to reduce the amount of pairs
of atoms to consider from N*N to something linear in N by removing pairs
that are farther away than a given cutoff radius.

This file implements this in a naive way: We first generate all N*N combinations,
and then trim down these candidates into a fixed number of pairs. Once that number
is decided, this procedure is jittable. The initial allocation is not.

The general data format consists of two index arrays, such that the indices in
`centers` contains the atom from which atom-pair vectors originate, while the
`others` contains the index which receives a given atom-pair vector.

The overall design of this is heavily inspired by jax-md by Samuel Schoenholz.

"""

from collections import namedtuple
from jax import jit, vmap
import jax.numpy as jnp
from jax.lax import stop_gradient, cond
from functools import partial
from typing import Callable

from glp import comms
from .periodic import displacement, get_heights
from .utils import boolean_mask_1d, cast, squared_distance

Neighbors = namedtuple(
    "Neighbors", ("centers", "others", "overflow", "reference_positions")
)


def neighbor_list(system, cutoff, skin, capacity_multiplier=1.25):
    # convenience interface

    allocate, update = quadratic_neighbor_list(
        system.cell, cutoff, skin, capacity_multiplier=capacity_multiplier
    )

    if hasattr(system, "padding_mask"):

        def _update(system, neighbors):
            neighbors = update(
                system.R,
                neighbors,
                new_cell=system.cell,
                padding_mask=system.padding_mask,
                force_update=system.updated,
            )
            return neighbors

        neighbors = allocate(system.R, padding_mask=system.padding_mask)
    else:

        def _update(system, neighbors):
            neighbors = update(system.R, neighbors, new_cell=system.cell)
            return neighbors

        neighbors = allocate(system.R)

    return neighbors, _update


def quadratic_neighbor_list(cell, cutoff, skin, capacity_multiplier=1.25, debug=False):
    """Toy implementation of neighborlist in pbc"""
    # todo: more sophisticated checks for varying cell
    # todo: use cell list

    assert capacity_multiplier >= 1.0

    cell = stop_gradient(cell)

    cutoff = cast(stop_gradient(cutoff))
    skin = cast(stop_gradient(skin))

    if cell is not None:

        def cell_too_small(new_cell):
            min_height = jnp.min(get_heights(new_cell))
            return cast(2) * (cutoff + skin) > min_height

    else:

        def cell_too_small(new_cell):
            return False

    if cell_too_small(cell):
        min_height = jnp.min(get_heights(cell))
        comms.warn(
            f"warning: total cutoff {(cutoff+skin):.2f} does not fit within simulation cell (needs to be < {0.5*min_height:.2f})"
        )
        comms.warn("this will yield incorrect results!")

    squared_cutoff = (cutoff + skin) ** cast(2.0)

    allowed_movement = (skin * cast(0.5)) ** cast(2.0)

    def need_update_fn(neighbors, new_positions, new_cell):
        # question: how to deal with changes in new_cell?
        # we will invalidate if atoms move too much, but not if
        # the new cell is too small for the cutoff...

        movement = make_squared_distance(new_cell)(
            neighbors.reference_positions, new_positions
        )
        max_movement = jnp.max(movement)

        # if we have an overflow, we don't need to update -- results are
        # already invalid. instead, we will simply return previous invalid
        # neighbors and hope someone up the chain catches it!
        should_update = (max_movement > allowed_movement) & (~neighbors.overflow)

        # if the cell is now *too small*, we need to update for sure to set overflow flag
        return should_update | cell_too_small(new_cell)

    def allocate_fn(positions, new_cell=None, padding_mask=None):
        # this is not jittable,
        # we're determining shapes

        if new_cell is None:
            new_cell = cell
        else:
            new_cell = stop_gradient(new_cell)

        N = positions.shape[0]

        centers, others, sq_distances, mask, hits = get_neighbors(
            positions,
            new_cell,
            make_squared_distance(new_cell),
            squared_cutoff,
            padding_mask=padding_mask,
        )

        size = int(hits.item() * capacity_multiplier + 1)
        centers, _ = boolean_mask_1d(centers, mask, size, N)
        others, overflow = boolean_mask_1d(others, mask, size, N)

        overflow = overflow | cell_too_small(new_cell)

        # print("done with neighbors=None branch")
        return Neighbors(centers, others, overflow, positions)

    def update_fn(positions, neighbors, new_cell=None, padding_mask=None, force_update=False):
        # this is jittable,
        # neighbors tells us all the shapes we need

        if new_cell is None:
            new_cell = cell
        else:
            new_cell = stop_gradient(new_cell)

        N = positions.shape[0]
        size = neighbors.centers.shape[0]

        def update(positions, cell, padding_mask):
            centers, others, sq_distances, mask, hits = get_neighbors(
                positions,
                cell,
                make_squared_distance(cell),
                squared_cutoff,
                padding_mask=padding_mask,
            )
            centers, _ = boolean_mask_1d(centers, mask, size, N)
            others, overflow = boolean_mask_1d(others, mask, size, N)

            overflow = overflow | cell_too_small(cell)

            return Neighbors(centers, others, overflow, positions)

        # if we need an update, call update(), else do a no-op and return input
        return cond(
            force_update | need_update_fn(neighbors, positions, new_cell),
            update,
            lambda a, b, c: neighbors,
            positions,
            new_cell,
            padding_mask,
        )

    if debug:
        return allocate_fn, update_fn, need_update_fn
    else:
        return allocate_fn, update_fn


def candidates_fn(n):
    candidates = jnp.arange(n)
    square = jnp.broadcast_to(candidates[None, :], (n, n))
    centers = jnp.reshape(jnp.transpose(square), (-1,))
    others = jnp.reshape(square, (-1,))
    return centers, others


def get_neighbors(positions, cell, square_distances, cutoff, padding_mask=None):
    centers, others = candidates_fn(positions.shape[0])
    sq_distances = square_distances(positions[centers], positions[others])

    mask = sq_distances <= cutoff
    mask = mask * (centers != others)  # remove self-interactions

    # facility to drop some positions from neighborlist
    if padding_mask is not None:
        mask = mask * padding_mask[centers]

    hits = jnp.sum(mask)

    return centers, others, sq_distances, mask, hits


def make_squared_distance(cell):
    return vmap(lambda Ra, Rb: squared_distance(displacement(cell, Ra, Rb)))
