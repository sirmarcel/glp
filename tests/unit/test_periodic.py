from unittest import TestCase

import numpy as np

import jax
import jax.numpy as jnp
from ase import Atoms

from glp import atoms_to_system
from glp.neighborlist import neighbor_list
from glp.graph import system_to_graph
from glp.system import System


class TestPeriodic(TestCase):
    def test_basic(self):
        # test a manually constructed example
        # this fails with jax-md displacement functions, i.e. when using
        # the following for periodic.displacement:
        # from jax_md import space
        # disp = space.periodic_general(cell, fractional_coordinates=False)[0]
        # return disp(Rb, Ra)

        atoms = Atoms(
            positions=[[1.0, 0.0, 0], [2, 3, 0]],
            cell=[[6, 0, 0], [2, 4, 0], [0, 0, 20]],
            pbc=True,
        )

        reference = atoms.get_all_distances(mic=True, vector=True)

        system = atoms_to_system(atoms)

        neighbors, update = neighbor_list(system, 2.0, 0.0)

        def mic(system, neighbors):
            return system_to_graph(system, neighbors).edges[0]

        jac_fn = jax.jacrev(mic, argnums=0, allow_int=True)

        fwd = mic(system, neighbors)

        np.testing.assert_allclose(fwd, reference[0, 1])

        jac = jac_fn(system, neighbors)

        assert jac.R[0, 0, 0] == jac.R[1, 0, 1] == jac.R[2, 0, 2]
        assert jac.R[0, 1, 0] == jac.R[1, 1, 1] == jac.R[2, 1, 2]

        np.testing.assert_allclose(jac.R[0, 0, 0], -1)
        np.testing.assert_allclose(jac.R[0, 1, 0], +1)

        np.testing.assert_allclose(jac.cell[0, 0, 0], 0, atol=1e-8)
        np.testing.assert_allclose(jac.cell[1, 1, 0], 0, atol=1e-8)
        np.testing.assert_allclose(jac.cell[2, 2, 0], 0, atol=1e-8)

        np.testing.assert_allclose(jac.cell[0, 0, 1], -1, atol=1e-8)
        np.testing.assert_allclose(jac.cell[1, 1, 1], -1, atol=1e-8)
        np.testing.assert_allclose(jac.cell[2, 2, 1], -1, atol=1e-8)
