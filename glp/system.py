from collections import namedtuple

from jax import numpy as jnp

from jax_md.space import periodic_general, free

System = namedtuple("System", ("R", "Z", "cell"))
UnfoldedSystem = namedtuple("System", ("R", "Z", "cell", "mask", "replica_idx"))


def atoms_to_system(atoms, dtype=jnp.float32):
    R = jnp.array(atoms.get_positions(), dtype=dtype)
    Z = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32) # we will infer this type
    cell = jnp.array(atoms.get_cell().array.T, dtype=dtype)
    return System(R, Z, cell)


def unfold_system(system, unfolding):
    from glp.unfold import unfold
    N = system.R.shape[0]

    wrapped, unfolded = unfold(system.R, system.cell, unfolding)
    all_R = jnp.concatenate((wrapped, unfolded), axis=0)
    all_idx = jnp.concatenate((jnp.arange(N), unfolding.replica_idx), axis=0)
    all_Z = system.Z[all_idx]

    mask = jnp.arange(all_R.shape[0]) < N

    return UnfoldedSystem(all_R, all_Z, None, mask, all_idx)


def to_displacement(system):
    if system.cell is not None:
        return periodic_general(system.cell, fractional_coordinates=False)[0]
    else:
        return free()[0]
