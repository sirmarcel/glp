import jax
import jax.numpy as jnp

from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list

from .calculator import Calculator
from .utils import strain_system, strain_graph, get_strain


def calculator(
    potential,
    system,
    skin=0.0,
    capacity_multiplier=1.25,
    stress_mode="strain_graph",  # allowed: strain_system, strain_graph, direct
):
    assert stress_mode in ["strain_system", "strain_graph", "direct"]
    cutoff = potential.cutoff

    state, update_neighbors = neighbor_list(
        system, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    if hasattr(potential, "update"):
        graph = system_to_graph(system, state)
        potential.update(graph)

    if stress_mode == "strain_system":

        def energy_fn(system, strain, state):
            state = update_neighbors(system, state)
            system = strain_system(system, strain)
            graph = system_to_graph(system, state)
            return jnp.sum(potential(graph)), state

        energies_and_derivatives_fn = jax.value_and_grad(
            energy_fn, argnums=(0, 1), allow_int=True, has_aux=True
        )

    elif stress_mode == "strain_graph":

        def energy_fn(system, strain, state):
            state = update_neighbors(system, state)
            graph = system_to_graph(system, state)
            graph = strain_graph(graph, strain)
            return jnp.sum(potential(graph)), state

        energies_and_derivatives_fn = jax.value_and_grad(
            energy_fn, argnums=(0, 1), allow_int=True, has_aux=True
        )

    elif stress_mode == "direct":

        def energy_fn(system, state):
            state = update_neighbors(system, state)
            graph = system_to_graph(system, state)
            return jnp.sum(potential(graph)), state

        energies_and_derivatives_fn = jax.value_and_grad(
            energy_fn, argnums=0, allow_int=True, has_aux=True
        )

    def calculator_fn(system, state, velocities=None, masses=None):
        if "strain" in stress_mode:
            strain = get_strain(dtype=system.R.dtype)
            energy_and_state, grads_and_stress = energies_and_derivatives_fn(
                system, strain, state
            )
            energy, state = energy_and_state
            grads, stress = grads_and_stress

        else:
            energy_and_state, grads = energies_and_derivatives_fn(system, state)
            energy, state = energy_and_state

            stress = jnp.einsum("ia,ib->ab", system.R, grads.R) + jnp.einsum(
                "aA,bA->ab", system.cell, grads.cell
            )

        forces = -grads.R

        output = {"energy": energy, "forces": forces, "stress": stress}

        if hasattr(potential, "update"):
            from glp.neighborlist import Neighbors

            graph = system_to_graph(system, state)
            state = Neighbors(
                state.centers,
                state.others,
                potential.check_overflow(graph) | state.overflow,
                state.reference_positions,
            )

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
