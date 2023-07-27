import jax
import jax.numpy as jnp

from glp import comms
from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list
from glp.utils import cast

from .calculator import Calculator


def calculator(
    potential,
    system,
    skin=0.0,
    capacity_multiplier=1.25,
    heat_flux=False,
    convective=True,
    fractional_mic=True,
):
    cutoff = potential.cutoff

    if heat_flux and cutoff < potential.effective_cutoff:
        comms.warn("heat flux will not be correct if cutoff < effective_cutoff")

    if not convective:
        comms.warn("convective flux disabled, will give incorrect flux")

    state, update_neighbors = neighbor_list(
        system, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    def energy_fn(graph):
        energies = potential(graph)
        return jnp.sum(energies), energies

    energies_and_derivatives_fn = jax.value_and_grad(
        energy_fn, allow_int=True, has_aux=True
    )

    def calculator_fn(system, state, velocities=None, masses=None):
        state = update_neighbors(system, state)

        graph = system_to_graph(system, state)

        energy, grads = energies_and_derivatives_fn(graph)
        energy_and_energies, grads = energies_and_derivatives_fn(graph)
        energy, energies = energy_and_energies
        forces_1 = jax.ops.segment_sum(
            grads.edges, graph.centers, graph.nodes.shape[0], indices_are_sorted=True
        )
        forces_2 = jax.ops.segment_sum(
            grads.edges, graph.others, graph.nodes.shape[0], indices_are_sorted=False
        )

        forces = forces_1 - forces_2

        stress = jnp.einsum("pa,pb->ab", graph.edges, grads.edges)

        output = {"energy": energy, "forces": forces, "stress": stress}

        if heat_flux:
            hf_potential = jnp.einsum(
                "pa,pb,pb->a", -graph.edges, grads.edges, velocities[graph.others]
            )
            output["heat_flux_potential"] = hf_potential

            if convective:
                energies_potential = energies

                # todo: this will fail obscurely due to index mismatch if masses is None. improve!
                energies_kinetic = cast(0.5) * jnp.einsum(
                    "i,ia,ia->i", masses, velocities, velocities
                )
                hf_convective = jnp.einsum(
                    "i,ia->a", energies_potential + energies_kinetic, velocities
                )
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
                capacity_multiplier=capacity_multiplier,
                heat_flux=heat_flux,
                convective=convective,
                fractional_mic=fractional_mic,
            ),
        ),
        state,
    )
