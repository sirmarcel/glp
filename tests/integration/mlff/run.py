from ase.io import read
from ase import units

from glp.instantiate import get_dynamics
from glp.dynamics import atoms_to_input

dct = {
    "potential": {"mlff": {"folder": "so3krates_SnSe/"}},
    "calculator": {"atom_pair": {}},
    "dynamics": {"verlet": {"dt": 1.0 * units.fs}},
}

inp = atoms_to_input(read("geometry.in"))
get_dynamics_fn = get_dynamics(**dct)

dynamics, state = get_dynamics_fn(*inp)

dynamics.step(state, None)
