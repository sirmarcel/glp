from jax import numpy as jnp
from importlib import import_module

from glp.ase import Calculator
from glp.instantiate import get_calculator

defaults_calculate = {
    "calculator": "atom_pair"
}


def calculator(**kwargs):
    get_calculator_fn = _get_calculator(kwargs)

    return Calculator(get_calculator_fn)

def _get_calculator(kwargs):
    assert "potential" in kwargs
    potential_dict = kwargs.pop("potential")

    assert "potential" in potential_dict
    potential_type = potential_dict.pop("potential")

    potential = {potential_type: potential_dict}

    if "calculator" in kwargs:
        calculator_dict = kwargs.pop("calculator")
    else:
        calculator_dict = {}

    calculator_dict = {**defaults_calculate, **calculator_dict}

    calculator_type = calculator_dict.pop("calculator")

    calculator = {calculator_type: calculator_dict}

    return get_calculator(potential, calculator)
