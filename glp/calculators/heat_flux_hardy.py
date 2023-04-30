import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from glp import comms
from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list
from glp.system import System, to_displacement
from glp.utils import cast

from .calculator import Calculator
from .utils import add_convective


def calculator(
    potential,
    system,
    skin=0.0,
    capacity_multiplier=1.25,
    convective=True,
):
    cutoff = potential.cutoff

    state, update_neighbors = neighbor_list(
        system, cutoff=cutoff, skin=skin, capacity_multiplier=capacity_multiplier
    )

    if not convective:
        comms.warn("convective flux disabled, will give incorrect flux")

    def energies_fn(system, state):
        return potential(system_to_graph(system, state))

    def calculator_fn(system, state, velocities, masses=None):
        state = update_neighbors(system, state)

        # needs to be here to support the case of changing cell
        # (which only occurs in tests)
        # todo: maybe drop this
        displacement_fn = to_displacement(system)

        N = system.Z.shape[0]

        def single_energy_fn(system, state, i):
            graph = system_to_graph(system, state)
            energies = potential(graph)
            return jnp.sum(
                energies
                * jax.nn.one_hot(
                    jnp.array([i], dtype=jnp.float32), N, dtype=jnp.float32
                ).flatten(),
            )

        grad_fn = jax.grad(single_energy_fn, argnums=0, allow_int=True)

        def to_scan(i, ignored):
            grad = grad_fn(system, state, i).R

            R = stop_gradient(system.R)

            Ri = jnp.tile(R[i], N).reshape(-1, 3)
            Rji = jax.vmap(displacement_fn)(Ri, R)

            hf = jnp.einsum("ja,jb,jb->a", Rji, grad, velocities)

            return i + cast(1), hf

        hfs = jax.lax.scan(to_scan, cast(0), None, length=N)[1]

        hf_potential = jnp.sum(hfs, axis=0)
        output = {"heat_flux_potential": hf_potential}

        if convective:
            output = add_convective(
                output, energies_fn(system, state), velocities, masses
            )
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
                convective=convective,
            ),
        ),
        state,
    )
