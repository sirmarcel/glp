from importlib import import_module

from specable.dicts import parse_dict

from .utils import str_to_dtype

def get_calculator(potential, calculator):
    calculator_fn = get_component("calculator", calculator)
    potential = get_component("potential", potential)

    return lambda system: calculator_fn(potential, system)


def get_dynamics(potential, calculator, dynamics):
    calculator_fn = get_calculator(potential, calculator)
    dynamics_fn = get_component("dynamics", dynamics)

    return lambda system, velocities, masses: dynamics_fn(
        system, velocities, masses, calculator_fn
    )


def get_component(kind, dct):
    assert kind in ["potential", "calculator", "dynamics"]

    handle, inner = parse_dict(dct, allow_stubs=True)

    if kind == "calculator":
        return get_calculator_fn(handle, inner)
    elif kind == "potential":
        return get_potential(handle, inner)
    elif kind == "dynamics":
        return get_dynamics_fn(handle, inner)


def get_calculator_fn(handle, inner):
    calc = import_module(f"glp.calculators.{handle}")
    return lambda potential, system: calc.calculator(potential, system, **inner)


def get_potential(handle, inner):

    if handle == "mlff":
        from mlff.mdx import MLFFPotential

        add_shift = inner.get("add_shift", False)
        dtype = str_to_dtype(inner.get("dtype", "float32"))

        potential = MLFFPotential.create_from_ckpt_dir(ckpt_dir=inner["folder"], dtype=dtype, add_shift=add_shift)

    elif handle in ["lennard_jones"]:
        from glp.potentials.lj import lennard_jones

        potential = lennard_jones(**inner)

    else:
        raise ValueError(f"unknown potential {handle}")

    return potential


def get_dynamics_fn(handle, inner):
    module = import_module(f"glp.dynamics")
    dynamics = getattr(module, handle)
    return lambda system, velocities, masses, get_calculator: dynamics(
        system, velocities, masses, get_calculator, **inner
    )
