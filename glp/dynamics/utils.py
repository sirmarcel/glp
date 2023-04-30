import jax.numpy as jnp
from collections import namedtuple

from glp.system import System

# phase space point: Positions, Momenta
Point = namedtuple("Point", ("R", "P"))

# MDState
MDState = namedtuple("MDState", ("point", "results", "calc_state", "overflow"))

Dynamics = namedtuple("Dynamics", ("step", "update"))


def to_velocities(momenta, masses):
    return momenta / masses[:, None]


def to_momenta(velocities, masses):
    return velocities * masses[:, None]


def to_system(point, system):
    return System(point.R, system.Z, system.cell)


def update(point, R=None, P=None):
    # get a new point
    if R is not None and P is None:
        return Point(R, point.P)
    elif P is not None and R is None:
        return Point(point.R, P)
    else:
        return Point(R, P)


def atoms_to_input(atoms, dtype=jnp.float32):
    from glp import atoms_to_system

    system = atoms_to_system(atoms, dtype=dtype)
    masses = jnp.array(atoms.get_masses(), dtype=dtype)
    velocities = jnp.array(atoms.get_velocities(), dtype=dtype)

    return system, velocities, masses
