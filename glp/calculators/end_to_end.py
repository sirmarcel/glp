import jax
import jax.numpy as jnp

from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list

from .calculator import Calculator
from .utils import strain_graph, get_strain


def calculator(
    potential,
    system,
    skin=0.0,
    capacity_multiplier=1.25,
):
    cutoff = potential.cutoff

    state, update_neighbors = neighbor_list(
        system, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    def energy_fn(system, strain, state):
        graph = system_to_graph(system, state)
        graph = strain_graph(graph, strain)
        return jnp.sum(potential(graph))

    energies_and_derivatives_fn = jax.value_and_grad(
        energy_fn, argnums=(0, 1), allow_int=True
    )

    def calculator_fn(system, state, velocities=None, masses=None):
        state = update_neighbors(system, state)

        strain = get_strain(dtype=system.R.dtype)
        energy, grads_and_stress = energies_and_derivatives_fn(
            system, strain, state
        )
        grads, stress = grads_and_stress

        forces = -grads.R

        output = {"energy": energy, "forces": forces, "stress": stress}

        return output, state

    return (
        Calculator(
            calculator_fn,
            lambda system: calculator(
                potential,
                system,
                skin=skin,
                capacity_multiplier=capacity_multiplier,
                stress_mode=stress_mode,
            ),
        ),
        state,
    )
