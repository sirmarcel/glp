from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from .utils import cast
from .periodic import displacement

Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others", "mask"))


def system_to_graph(system, neighbors):
    # neighbors are an *updated* neighborlist
    # question: how do we treat batching?

    positions = system.R
    nodes = system.Z

    edges = jax.vmap(partial(displacement, system.cell))(
        positions[neighbors.centers], positions[neighbors.others]
    )

    mask = neighbors.centers != positions.shape[0]

    return Graph(edges, nodes, neighbors.centers, neighbors.others, mask)
