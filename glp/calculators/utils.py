import jax
from jax.lax import stop_gradient
import jax.numpy as jnp

from glp.system import System, UnfoldedSystem
from glp.graph import Graph
from glp.utils import cast
from glp.periodic import displacement_real


def strain_system(system, strain):
    strained_R = system.R + jnp.einsum("ab,ib->ia", strain, system.R)
    strained_cell = system.cell + jnp.einsum("ab,bA->aA", strain, system.cell)
    return System(strained_R, system.Z, strained_cell)


def strain_unfolded_system(system, strain):
    strained_R = system.R + jnp.einsum("ab,ib->ia", strain, system.R)
    return UnfoldedSystem(
        strained_R, system.Z, system.cell, system.mask, system.replica_idx
    )


def strain_graph(graph, strain):
    strained_edges = graph.edges + jnp.einsum("ab,ib->ia", strain, graph.edges)
    return Graph(strained_edges, graph.nodes, graph.centers, graph.others, graph.mask)


def get_strain(dtype=jnp.float32):
    strain = jnp.zeros((3, 3), dtype=dtype)
    return cast(0.5) * (strain + strain.T)


def system_to_graph_without_mic(system, neighbors):
    positions = stop_gradient(system.R)
    cell = stop_gradient(system.cell)

    displacement_fn = displacement_real(cell)

    def offset_fn(Ri, Rj):
        real = Rj - Ri
        mic = displacement_fn(Rj, Ri)
        offset = jnp.linalg.solve(cell, mic - real)  # offset applied to Rj
        return offset

    offsets = jax.vmap(offset_fn)(
        positions[neighbors.centers], positions[neighbors.others]
    )
    cell_offsets = jnp.einsum("aA,pA->pa", system.cell, offsets)
    edges = system.R[neighbors.others] + cell_offsets - system.R[neighbors.centers]

    mask = neighbors.centers != positions.shape[0]

    return Graph(edges, system.Z, neighbors.centers, neighbors.others, mask)


def add_convective(output, energies_potential, velocities, masses):
    if masses is None:
        raise ValueError("calculator needs masses to compute convective flux")

    energies_kinetic = cast(0.5) * jnp.einsum(
        "i,ia,ia->i", masses, velocities, velocities
    )
    hf_convective = jnp.einsum(
        "i,ia->a", energies_potential + energies_kinetic, velocities
    )
    output["heat_flux"] = output["heat_flux_potential"] + hf_convective
    output["heat_flux_convective"] = hf_convective

    return output
