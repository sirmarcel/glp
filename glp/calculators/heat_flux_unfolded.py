import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from functools import partial
from collections import namedtuple

from glp import comms
from glp.system import System, unfold_system, UnfoldedSystem
from glp.graph import system_to_graph, Graph
from glp.neighborlist import neighbor_list
from glp.unfold import unfolder
from glp.utils import cast

from .calculator import Calculator

State = namedtuple("State", ("neighbors", "unfolding", "overflow"))


def calculator(
    potential,
    system,
    skin=0.0,
    skin_unfolder=0.5,
    capacity_multiplier=1.25,
    convective=True,
):
    cutoff = potential.cutoff
    cutoff_unfolder = potential.effective_cutoff

    unfolding, update_unfolding = unfolder(system, cutoff_unfolder, skin_unfolder)
    big = unfold_system(system, unfolding)
    neighbors, update_neighbors = neighbor_list(
        big, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    if not convective:
        comms.warn("heat flux without convective term will be incorrect")

    state = State(neighbors, unfolding, jnp.array(False))

    # todo: consider graph only approach
    def energy_fn(big, state):
        graph = system_to_graph(big, state.neighbors)
        energies = potential(graph) * big.mask
        return jnp.sum(energies), energies

    energy_and_derivatives_fn = jax.value_and_grad(
        energy_fn, argnums=0, allow_int=True, has_aux=True
    )

    def barycenter_fn(big, state, r_aux):
        graph = system_to_graph(big, state.neighbors)
        energies = potential(graph) * big.mask
        barycenter = energies[:, None] * r_aux
        return jnp.sum(barycenter, axis=0)

    def calculator_fn(system, state, velocities, masses=None):
        unfolding = update_unfolding(system, state.unfolding)

        big = unfold_system(system, unfolding)
        neighbors = update_neighbors(big, state.neighbors)

        state = State(
            neighbors, state.unfolding, unfolding.overflow | neighbors.overflow
        )

        energy_and_energies, grads = energy_and_derivatives_fn(big, state)
        energy, energies = energy_and_energies
        forces = jax.ops.segment_sum(-grads.R, big.replica_idx, system.Z.shape[0])
        stress = jnp.einsum("ia,ib->ab", big.R, grads.R)

        # heat flux
        unfolded_velocities = velocities[big.replica_idx]
        r_aux = stop_gradient(big.R)

        _, term_1 = jax.jvp(
            lambda R: barycenter_fn(
                UnfoldedSystem(R, big.Z, big.cell, big.mask, big.replica_idx, big.padding_mask, big.updated),
                state,
                r_aux,
            ),
            (big.R,),
            (unfolded_velocities,),
        )

        term_2 = jnp.sum(
            jnp.sum(grads.R * unfolded_velocities, axis=1)[:, None] * r_aux,
            axis=0,
        )
        hf_potential = term_1 - term_2

        output = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "heat_flux_potential": hf_potential,
        }

        if convective:
            # can't use the helper from utils --
            # we need to use unfolded velocities for the potential part

            if masses is None:
                raise ValueError("calculator needs masses to compute convective flux")

            energies_kinetic = cast(0.5) * jnp.einsum(
                "i,ia,ia->i", masses, velocities, velocities
            )
            hf_convective_kinetic = jnp.einsum("i,ia->a", energies_kinetic, velocities)
            hf_convective_potential = jnp.einsum(
                "i,ia->a", energies, unfolded_velocities
            )
            hf_convective = hf_convective_potential + hf_convective_kinetic
            output["heat_flux"] = hf_potential + hf_convective
            output["heat_flux_convective"] = hf_convective

        else:
            output["heat_flux"] = hf_potential

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
                convective=convective,
            ),
        ),
        state,
    )
