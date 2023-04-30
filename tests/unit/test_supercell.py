from unittest import TestCase

import jax

import numpy as np

from ase.build import bulk
from ase.calculators.lj import LennardJones

from glp.system import atoms_to_system
from glp.potentials import lennard_jones
from glp.calculators import supercell


class TestLJ(TestCase):
    def setUp(self):
        self.sigma = 2.0
        self.epsilon = 1.5
        self.rc = 5.5
        self.ro = 4.5

        self.calculator = LennardJones(
            epsilon=self.epsilon, sigma=self.sigma, rc=self.rc, ro=self.ro, smooth=True
        )

        self.potential = lennard_jones(
            epsilon=self.epsilon, sigma=self.sigma, cutoff=self.rc, onset=self.ro
        )

    def test_atom_pair(self):

        atoms = bulk("Ar", cubic=False) * [3, 3, 3]
        print(atoms.get_cell())
        print(len(atoms))

        atoms.rattle(stdev=0.2, seed=1)

        system = atoms_to_system(atoms)

        calculator, state = supercell.calculator(
            self.potential, system, n_replicas=2
        )

        calculate = jax.jit(calculator.calculate)
        res, state = calculate(system, state, velocities=None)

        atoms.calc = self.calculator
        atoms.get_potential_energy()

        np.testing.assert_allclose(
            res["energy"], atoms.get_potential_energy(), atol=1e-5
        )
        np.testing.assert_allclose(res["forces"], atoms.get_forces(), atol=1e-5)
        np.testing.assert_allclose(
            res["stress"], atoms.get_stress(voigt=False) * atoms.get_volume(), atol=5e-4
        )
