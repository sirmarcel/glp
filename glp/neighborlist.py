"""neighborlist.py

The role of a neighborlist implementation is to reduce the amount of pairs
of atoms to consider from N*N to something linear in N by removing pairs
that are farther away than a given cutoff radius.

This file implements this in two ways: 
- N-square: We first generate all N*N combinations, and then trim down these
            candidates into a fixed number of pairs. Once that number is
            decided, this procedure is jittable. The initial allocation is not.

- Cell-list: We first divide the simulation cell into a grid of bins, and 
                assign each atom to a bin. Then, for each atom, we only consider
                the atoms in the same bin and its neighbors. This is jittable. 
                The initial allocation is not.

The cell-list implementation is more efficient for large systems, but it is
more complex and requires the simulation cell to be periodic. The N-square
implementation is simpler and can be used for non-periodic systems.

The general data format consists of two index arrays, such that the indices in
`centers` contains the atom from which atom-pair vectors originate, while the
`others` contains the index which receives a given atom-pair vector.

The overall design of this is heavily inspired by jax-md by Samuel Schoenholz.

"""

from collections import namedtuple
import jax
from jax import vmap, ops
import jax.numpy as jnp
from jax.lax import stop_gradient, cond, iota
import numpy as np

from glp import comms
from .periodic import displacement, get_heights, wrap, inverse
from .utils import boolean_mask_1d, cast, squared_distance

Neighbors = namedtuple(
    "Neighbors", ("centers", "others", "overflow", "reference_positions", "cell_list")
)

CellList = namedtuple(
    "CellList", ("id", "reallocate", "capacity", "size", "bins_per_side")
)

def neighbor_list(system, cutoff, skin, capacity_multiplier=1.25):
    # convenience interface: we often don't need explicit access to an allocate_fn, since
    #     the neighborlist setup takes that role (see for instance all the calculators)
    #     so this just does the allocation and gives us a jittable update function back

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


def quadratic_neighbor_list(cell, cutoff, skin, capacity_multiplier=1.25, use_cell_list=False, debug=False):
    """Implementation of neighborlist in pbc using cell list."""

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

    cutoff = (cutoff + skin)

    allowed_movement = (skin * cast(0.5)) ** cast(2.0)

    if cell is not None and use_cell_list:
        if jnp.all(jnp.any(cutoff < get_heights(cell) / 3., axis=0)):
            cl_allocate, cl_update = cell_list(cell, cutoff)
        else:
            cl_allocate = cl_update = None
    else:
        cl_allocate = cl_update = None
        
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
        # Check if we are using cell list
        cl = cl_allocate(positions)  if cl_allocate is not None else None
        centers, others, sq_distances, mask, hits = get_neighbors(
            positions,
            make_squared_distance(new_cell),
            cutoff,
            padding_mask=padding_mask,
            cl=cl
        )
        size = int(hits.item() * capacity_multiplier + 1)
        centers, _ = boolean_mask_1d(centers, mask, size, N)
        others, overflow = boolean_mask_1d(others, mask, size, N)
        overflow = overflow | cell_too_small(new_cell)

        return Neighbors(centers, others, overflow, positions, cl)
    
    def update_fn(positions, neighbors, new_cell=None, padding_mask=None, force_update=False):
        # this is jittable,
        # neighbors tells us all the shapes we need
        if new_cell is None:
            new_cell = cell
        else:
            new_cell = stop_gradient(new_cell)
        
        
        N = positions.shape[0]
        dim = positions.shape[1]
        size = neighbors.centers.shape[0]
        def update(positions, cell, padding_mask, cl=neighbors.cell_list):
            # Check if we are using cell list
            cl = cl_update(positions, cl, new_cell)  if cl_update is not None else None
            centers, others, sq_distances, mask, hits = get_neighbors(
                positions,
                make_squared_distance(cell),
                cutoff,
                padding_mask=padding_mask,
                cl = cl
            )
            centers, _ = boolean_mask_1d(centers, mask, size, N)
            others, overflow = boolean_mask_1d(others, mask, size, N)
            overflow = overflow | cell_too_small(cell)

            return Neighbors(centers, others, overflow, positions, cl)

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

def cell_list_candidate_fn(bin_id, N, dim):
    """Get the candidates for the cell list neighbor list. It is implemented following the jax-md implementation."""
    idx = bin_id
    bin_idx = [idx] * (dim**3)
    # Get the neighboring bins from the current bin
    for i, dindex in enumerate(neighboring_bins(dim)):
        bin_idx[i] = shift_array(idx, dindex)

    bin_idx = jnp.concatenate(bin_idx, axis=-2)
    bin_idx = bin_idx[..., jnp.newaxis, :, :]
    bin_idx = jnp.broadcast_to(bin_idx, idx.shape[:-1] + bin_idx.shape[-2:])
    def copy_values_from_bin(value, bin_value, bin_id):
        scatter_indices = jnp.reshape(bin_id, (-1,))
        bin_value = jnp.reshape(bin_value, (-1,) + bin_value.shape[-2:])
        return value.at[scatter_indices].set(bin_value)
    neighbor_idx = jnp.zeros((N + 1,) + bin_idx.shape[-2:], jnp.int32)
    neighbor_idx = copy_values_from_bin(neighbor_idx, bin_idx, idx)
    # Reshape the neighbor_idx to get the centers and others
    others = jnp.reshape(neighbor_idx[:-1, :, 0], (-1,))
    centers = jnp.repeat(jnp.arange(0, N), others.shape[0] // N)
    return centers, others

def get_neighbors(positions, square_distances, cutoff, padding_mask=None, cl=None):
    N, dim = positions.shape
    if cl is not None:
        centers, others = cell_list_candidate_fn(cl.id, N, dim)
    else:
        centers, others = candidates_fn(N)

    
    sq_distances = square_distances(positions[centers], positions[others])
    mask = sq_distances <= (cutoff**2)
    mask = mask * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions
    if padding_mask is not None:
        mask = mask * padding_mask[centers]
        mask = mask * padding_mask[others]

    hits = jnp.sum(mask)
    return centers, others, sq_distances, mask, hits


def make_squared_distance(cell):
    return vmap(lambda Ra, Rb: squared_distance(displacement(cell, Ra, Rb)))


def cell_list(cell, cutoff, buffer_size_multiplier=1.25, bin_size_multiplier=1):
    """Implementation of cell list neighborlist in pbc."""

    cell = jnp.array(cell)
    cutoff *= bin_size_multiplier
    def allocate_fn(positions, extra_capacity=0):
        # This function is not jittable, we're determining shapes
        N = positions.shape[0]
        _, bin_size, bins_per_side, bin_count = bin_dimensions(cell, cutoff)
        bin_capacity = estimate_bin_capacity(positions, cell, bin_size,
                                            buffer_size_multiplier)
        # Computing the maximum number of atoms per bin (capacity)
        # extra_capacity is used to increase the capacity of the bins to avoid reallocation
        bin_capacity += extra_capacity
        
        overflow = False
        bin_id = N * jnp.ones((bin_count * bin_capacity, 1), dtype=jnp.int32)

        # Compute the hash of each particle to assign it to a bin
        hash_multipliers = compute_hash_constants(bins_per_side)
        particle_id = iota(jnp.int32, N)
        indices = jnp.array(jnp.floor(positions @ inverse(bin_size).T), dtype=jnp.int32)
        # Some particles are in the edge and might have negative indices or larger than bins_per_side
        # We need to correct them wrapping into bin per side vector
        indices = wrap(jnp.diag(bins_per_side), indices).astype(jnp.int32)
        hashes = jnp.sum(indices * hash_multipliers, axis=1, dtype=jnp.int32)

        sort_map = jnp.argsort(hashes)
        sorted_hash = hashes[sort_map]
        sorted_id = particle_id[sort_map]

        sorted_bin_id = jnp.mod(iota(jnp.int32, N), bin_capacity)
        sorted_bin_id = sorted_hash * bin_capacity + sorted_bin_id
        sorted_id = jnp.reshape(sorted_id, (N, 1))
        bin_id = bin_id.at[sorted_bin_id].set(sorted_id)
        bin_id = unflatten_bin_buffer(bin_id, bins_per_side)
        occupancy = ops.segment_sum(jnp.ones_like(hashes), hashes, bin_count)
        max_occupancy = jnp.max(occupancy)
        overflow = overflow | (max_occupancy > bin_capacity)
        return CellList(bin_id, overflow, bin_capacity, bin_size, bins_per_side)
    
    def update_fn(positions, old_cell_list, new_cell):
        # this is jittable,
        # CellList tells us all the shapes we need
        # If bin size is lower than cutoff, we need to reallocate cell list
        # Rellocate is not jitable, we're changing shapes
        # So, after each update we need to check if we need to reallocate
        N = positions.shape[0]

        bin_size = jnp.where(old_cell_list.bins_per_side != 0, cell / old_cell_list.bins_per_side, cell)
        max_occupancy = estimate_bin_capacity(positions, new_cell, bin_size, 1)
        # Checking if update or reallocate
        reallocate = jnp.all(get_heights(bin_size).any() >= cutoff) & (max_occupancy <= old_cell_list.capacity)

        def update(positions, old_cell_list):
            hash_multipliers = compute_hash_constants(old_cell_list.bins_per_side)
            indices = jnp.array(jnp.floor(positions @ inverse(bin_size).T), dtype=jnp.int32)
            # Some particles are in the edge and might have negative indices or larger than bins_per_side
            # We need to correct them wrapping into bin per side vector
            indices = wrap(jnp.diag(old_cell_list.bins_per_side), indices).astype(jnp.int32)
            
            hashes = jnp.sum(indices * hash_multipliers, axis=1, dtype=jnp.int32)
            particle_id = iota(jnp.int32, N)
            sort_map = jnp.argsort(hashes)
            sorted_hash = hashes[sort_map]
            sorted_id = particle_id[sort_map]

            sorted_bin_id = jnp.mod(iota(jnp.int32, N), old_cell_list.capacity)
            sorted_bin_id = sorted_hash * old_cell_list.capacity + sorted_bin_id
            sorted_id = jnp.reshape(sorted_id, (N, 1))
            bin_id = N * jnp.ones((old_cell_list.id.reshape(-1, 1).shape), dtype=jnp.int32)
            bin_id = bin_id.at[sorted_bin_id].set(sorted_id)

            # This is not jitable, we're changing shapes. It's a fix for the unflatten_bin_buffer.
            
            bin_id = jax.pure_callback(unflatten_bin_buffer, old_cell_list.id, bin_id, old_cell_list.bins_per_side)
            return CellList(bin_id, old_cell_list.reallocate, old_cell_list.capacity, bin_size, old_cell_list.bins_per_side)
        
        # In case bin size is lower than cutoff, we need to reallocate 
        def need_reallocate(_ ,old_cell_list):
            return CellList(old_cell_list.id, True, old_cell_list.capacity, bin_size, old_cell_list.bins_per_side)
        return cond(reallocate, need_reallocate, update, positions, old_cell_list)
    return allocate_fn, update_fn 


def estimate_bin_capacity(positions, cell, bin_size, buffer_size_multiplier):
    minimum_bin_size = jnp.min(get_heights(bin_size))
    bin_capacity = jnp.max(count_bin_filling(positions, cell, minimum_bin_size))
    return (bin_capacity * buffer_size_multiplier).astype(jnp.int32)

def bin_dimensions(cell, cutoff):
    """Compute the number of bins-per-side and total number of bins in a box."""
    # Considering cell is 3x3 array
    # Transform into reciprocal space to get the cell size whatever cell is
    face_dist = get_heights(cell)
    bins_per_side = jnp.floor(face_dist / cutoff).astype(jnp.int32)
    bin_size = jnp.where(bins_per_side != 0, cell / bins_per_side, cell)
    bin_count = jnp.prod(bins_per_side)
    return cell, bin_size, bins_per_side, bin_count.astype(jnp.int32)

    
def count_bin_filling(position, cell, minimum_bin_size):
    """
    Counts the number of particles per-bin in a spatial partitioning scheme.
    """
    cell, bin_size, bins_per_side, _ = bin_dimensions(cell, minimum_bin_size)
    hash_multipliers = compute_hash_constants(bins_per_side)
    particle_index = jnp.array(jnp.floor(jnp.dot(position, inverse(bin_size.T).T)), dtype=jnp.int32)
    particle_hash = jnp.sum(particle_index * hash_multipliers, axis=1)
    filling = jnp.zeros_like(particle_hash, dtype=jnp.int32)
    filling = filling.at[particle_hash].add(1)
    return filling

def compute_hash_constants(bins_per_side):
    """Compute the hash constants for a given number of bins per side."""
    one = jnp.array([1])
    bins_per_side = jnp.concatenate((one, bins_per_side[:-1]), axis=0)
    return jnp.array(jnp.cumprod(bins_per_side), dtype=jnp.int32)

def unflatten_bin_buffer(arr, bins_per_side):
    """Unflatten the bin buffer to get the bin_id."""
    bins_per_side = tuple(bins_per_side)
    return jnp.reshape(arr, bins_per_side + (-1,) + arr.shape[1:])

def neighboring_bins(dimension):
  """Generate the indices of the neighboring bins of a given bin"""
  for dindex in np.ndindex(*([3] * dimension)):
    yield jnp.array(dindex) - 1



def shift_array(arr, dindex):
    """Shift an array by a given index. It is used for the bin's neighbor list."""
    dx, dy, dz = tuple(dindex) + (0,) * (3 - len(dindex))
    arr = cond(dx < 0,
               lambda x: jnp.concatenate((x[1:], x[:1])),
               lambda x: cond(dx > 0,
                              lambda x: jnp.concatenate((x[-1:], x[:-1])),
                              lambda x: x, arr),
               arr)

    arr = cond(dy < 0,
               lambda x: jnp.concatenate((x[:, 1:], x[:, :1]), axis=1),
               lambda x: cond(dy > 0,
                              lambda x: jnp.concatenate((x[:, -1:], x[:, :-1]), axis=1),
                              lambda x: x, arr),
               arr)

    arr = cond(dz < 0,
               lambda x: jnp.concatenate((x[:, :, 1:], x[:, :, :1]), axis=2),
               lambda x: cond(dz > 0,
                              lambda x: jnp.concatenate((x[:, :, -1:], x[:, :, :-1]), axis=2),
                              lambda x: x, arr),
               arr)

    return arr

# I had to include it for reallocate cell list, calling cell_list allocate not jitable
def check_reallocation_cell_list(cell_list, allocate_fn, positions):
    if cell_list.reallocate:
        return allocate_fn(positions)
    return cell_list