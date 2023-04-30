from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from .system import to_displacement
from .utils import cast

Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others", "mask"))


def system_to_graph(system, neighbors):
    # neighbors are an *updated* neighborlist
    # question: how do we treat batching?

    positions = system.R
    nodes = system.Z

    displacement_fn = to_displacement(system)

    edges = jax.vmap(displacement_fn)(
        positions[neighbors.others], positions[neighbors.centers]
    )

    mask = neighbors.centers != positions.shape[0]

    return Graph(edges, nodes, neighbors.centers, neighbors.others, mask)
