import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple

from glp.system import System, unfold_system
from glp.graph import system_to_graph, Graph
from glp.neighborlist import neighbor_list
from glp.unfold import unfolder

from .calculator import Calculator
from .utils import strain_system, get_strain, strain_unfolded_system

State = namedtuple("State", ("neighbors", "unfolding", "overflow"))


def calculator(
    potential,
    system,
    skin=0.0,
    skin_unfolder=0.5,
    capacity_multiplier=1.25,
    stress_mode="direct_system",  # allowed: direct_system, direct_unfolded, strain_unfolded
):
    assert stress_mode in ["direct_system", "direct_unfolded", "strain_unfolded"]

    cutoff = potential.cutoff
    cutoff_unfolder = potential.effective_cutoff

    unfolding, check_unfolding = unfolder(system, cutoff_unfolder, skin_unfolder)
    big = unfold_system(system, unfolding)
    neighbors, update_neighbors = neighbor_list(
        big, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    state = State(neighbors, unfolding, jnp.array(False))

    if stress_mode == "direct_system":

        def energy_fn(system, state):
            overflow_unfolding = check_unfolding(system, state.unfolding)

            big = unfold_system(system, state.unfolding)
            neighbors = update_neighbors(big, state.neighbors)

            state = State(
                neighbors, state.unfolding, overflow_unfolding | neighbors.overflow
            )

            graph = system_to_graph(big, state.neighbors)

            return jnp.sum(potential(graph) * big.mask), state

        energies_and_derivatives_fn = jax.value_and_grad(
            energy_fn, argnums=0, allow_int=True, has_aux=True
        )

    elif stress_mode == "direct_unfolded":

        def energy_fn(big, state):
            graph = system_to_graph(big, state.neighbors)

            return jnp.sum(potential(graph) * big.mask)

        energies_and_derivatives_fn = jax.value_and_grad(
            energy_fn, argnums=0, allow_int=True, has_aux=False
        )

    elif stress_mode == "strain_unfolded":

        def energy_fn(big, strain, state):
            big = strain_unfolded_system(big, strain)
            graph = system_to_graph(big, state.neighbors)

            return jnp.sum(potential(graph) * big.mask)

        energies_and_derivatives_fn = jax.value_and_grad(
            energy_fn, argnums=(0, 1), allow_int=True, has_aux=False
        )

    def calculator_fn(system, state, velocities=None, masses=None):
        if stress_mode == "direct_system":
            energy_and_state, grads = energies_and_derivatives_fn(system, state)
            energy, state = energy_and_state

            forces = -grads.R

            stress = jnp.einsum("ia,ib->ab", system.R, grads.R) + jnp.einsum(
                "aA,bA->ab", system.cell, grads.cell
            )

        elif stress_mode == "direct_unfolded":
            overflow_unfolding = check_unfolding(system, state.unfolding)

            big = unfold_system(system, state.unfolding)
            neighbors = update_neighbors(big, state.neighbors)

            state = State(
                neighbors, state.unfolding, overflow_unfolding | neighbors.overflow
            )

            energy, grads = energies_and_derivatives_fn(big, state)

            forces = jax.ops.segment_sum(
                -grads.R, big.replica_idx, num_segments=system.Z.shape[0]
            )
            stress = jnp.einsum("ia,ib->ab", big.R, grads.R)

        elif stress_mode == "strain_unfolded":
            overflow_unfolding = check_unfolding(system, state.unfolding)

            big = unfold_system(system, unfolding)
            neighbors = update_neighbors(big, state.neighbors)

            state = State(
                neighbors, state.unfolding, overflow_unfolding | neighbors.overflow
            )

            strain = get_strain(dtype=system.R.dtype)
            energy, grads_and_stress = energies_and_derivatives_fn(big, strain, state)
            grads, stress = grads_and_stress

            forces = jax.ops.segment_sum(
                -grads.R, big.replica_idx, num_segments=system.Z.shape[0]
            )

        output = {"energy": energy, "forces": forces, "stress": stress}

        return output, state

    return (
        Calculator(
            calculator_fn,
            lambda system: calculator(
                potential,
                system,
                skin=skin,
                skin_unfolder=skin_unfolder,
                capacity_multiplier=capacity_multiplier,
                stress_mode=stress_mode,
            ),
        ),
        state,
    )
