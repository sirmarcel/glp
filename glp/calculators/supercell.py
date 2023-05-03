import jax
import jax.numpy as jnp
from functools import partial

from glp.system import System
from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list
from .utils import strain_graph, get_strain
from .calculator import Calculator


def calculator(potential, system, skin=0.0, n_replicas=2, capacity_multiplier=1.25):
    cutoff = potential.cutoff

    # TODO: shall auto determination of replicas be an option or not?
    multiplier = jnp.array(n_replicas**3, dtype=system.R.dtype)

    big = make_supercell(system, n_replicas)

    init_state, update_neighbors = neighbor_list(
        big, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    def energy_fn(sys, strain, state):
        big_sys = make_supercell(sys, n_replicas)
        new_state = update_neighbors(big_sys, state)
        graph = system_to_graph(big_sys, new_state)
        strained_graph = strain_graph(graph, strain)

        return jnp.sum(potential(strained_graph)) / multiplier, new_state

    energy_and_derivatives_fn = jax.value_and_grad(
        energy_fn, allow_int=True, argnums=(0, 1), has_aux=True
    )

    def calculator_fn(sys, state, velocities=None, masses=None):
        strain = get_strain(dtype=sys.R.dtype)
        energy_and_state, grads_and_stress = energy_and_derivatives_fn(
            sys, strain, state
        )

        energy, state = energy_and_state
        grads, stress = grads_and_stress

        forces = -grads.R

        output = {"energy": energy, "forces": forces, "stress": stress}

        return output, state

    return (
        Calculator(
            calculator_fn,
            lambda sys: calculator(
                potential,
                sys,
                skin=skin,
                capacity_multiplier=capacity_multiplier,
                n_replicas=n_replicas,
            ),
        ),
        init_state,
    )


def get_offsets(n_replicas):
    a = jnp.arange(n_replicas)
    return jnp.array(jnp.meshgrid(a, a, a)).T.reshape(-1, 3)


def replicate(positions, cell, offsets):
    return positions + jnp.einsum("aA,iA->ia", cell, offsets)


@partial(jax.jit, static_argnums=1)
def make_supercell(system, n_replicas):
    offsets = get_offsets(n_replicas)

    all_replicas = jax.vmap(lambda o: replicate(system.R, system.cell, o[None, :]))(
        offsets
    )
    new_positions = all_replicas.reshape(-1, 3)
    new_cell = system.cell * n_replicas
    new_charges = jnp.tile(system.Z, n_replicas**3)

    return System(new_positions, new_charges, new_cell)
