import jax
import jax.numpy as jnp
from jax_md.space import distance

from glp.neighborlist import quadratic_neighbor_list
from glp.system import atoms_to_system
from glp.graph import system_to_graph
from glp.utils import cast
from .potential import Potential


def lennard_jones(sigma=2.0, epsilon=1.5, cutoff=10.0, onset=6.0):
    # we assume double counting, so 4*epsilon/2 is the prefactor
    factor = cast(2 * epsilon)
    sigma = cast(sigma)
    cutoff2 = cast(cutoff**2)
    onset2 = cast(onset**2)
    zero = cast(0.0)
    one = cast(1.0)

    def pairwise_energy_fn(dr):
        inverse_r = sigma / dr
        inverse_r6 = inverse_r ** cast(6.0)
        inverse_r12 = inverse_r6 * inverse_r6

        return factor * (inverse_r12 - inverse_r6)

    def cutoff_fn(dr):
        # inspired by jax-md, which in turns uses HOOMD-BLUE

        distance2 = dr ** cast(2.0)

        # in between onset and infinity:
        # either our mollifier or zero
        after_onset = jnp.where(
            distance2 < cutoff2,
            (cutoff2 - distance2) ** cast(2.0)
            * (cutoff2 + cast(2.0) * distance2 - cast(3.0) * onset2)
            / (cutoff2 - onset2) ** cast(3.0),
            zero,
        )

        # do nothing before onset, then mollify
        return jnp.where(
            distance2 < onset2,
            one,
            after_onset,
        )

    def pair_lj(dr):
        return cutoff_fn(dr) * pairwise_energy_fn(dr)

    def lennard_jones_fn(graph):
        distances = distance(graph.edges)
        contributions = jax.vmap(pair_lj)(distances)
        out = contributions * graph.mask
        # a = cast(0.5) * jax.ops.segment_sum(out, graph.centers, graph.nodes.shape[0], indices_are_sorted=True)
        # b = cast(0.5) * jax.ops.segment_sum(out, graph.others, graph.nodes.shape[0], indices_are_sorted=False)

        return jax.ops.segment_sum(out, graph.centers, graph.nodes.shape[0], indices_are_sorted=True)

    return Potential(lennard_jones_fn, cutoff)


def lennard_jones_jax_md(sigma=2.0, epsilon=1.5, cutoff=10.0, onset=6.0):
    from jax_md import energy, util

    sigma = util.f32(sigma)
    epsilon = util.f32(epsilon)
    cutoff = util.f32(cutoff)
    onset = util.f32(onset)

    def _pair_lennard_jones_argon(dr):
        return util.f32(0.5) * energy.lennard_jones(dr, sigma=sigma, epsilon=epsilon)

    pair_lennard_jones_argon = energy.multiplicative_isotropic_cutoff(
        _pair_lennard_jones_argon, r_onset=onset, r_cutoff=cutoff
    )

    def _lennard_jones(graph):
        distances = distance(graph.edges)
        contributions = jax.vmap(pair_lennard_jones_argon)(distances)
        out = contributions * graph.mask
        out = jax.ops.segment_sum(out, graph.centers, graph.nodes.shape[0])

        return out

    return _lennard_jones
