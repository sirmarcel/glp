from unittest import TestCase

import numpy as np

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp


from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs
from ase.calculators.lj import LennardJones
from ase.units import fs
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen

from glp.ase.calculator import Calculator

from glp.system import atoms_to_system
from glp.potentials import lennard_jones
from glp.calculators import atom_pair


class TestLJ(TestCase):
    def setUp(self):
        self.sigma = 3.4
        self.epsilon = 0.011
        self.rc = 10.0
        self.ro = 9.0

        self.calculator = LennardJones(
            epsilon=self.epsilon, sigma=self.sigma, rc=self.rc, ro=self.ro, smooth=True
        )

        self.potential = lennard_jones(
            epsilon=self.epsilon, sigma=self.sigma, cutoff=self.rc, onset=self.ro
        )

    def test_npt(self):
        from glp.periodic import get_heights

        gh = lambda x: get_heights(atoms_to_system(x).cell)

        dt = 1 * fs
        nsteps = 25
        temperature_K = 50
        pressure = 1

        atoms = bulk("Ar", cubic=False) * [9, 8, 8]
        atoms.set_cell(0.98 * atoms.get_cell(), scale_atoms=True)
        MaxwellBoltzmannDistribution(atoms, temperature_K=10)
        Stationary(atoms)
        atoms2 = atoms.copy()

        system = atoms_to_system(atoms2, dtype=jnp.float64)

        get_calculator = lambda system: atom_pair.calculator(
            self.potential, system, skin=0.1, heat_flux=False
        )

        calc = Calculator(get_calculator, atoms, dtype=jnp.float64)

        atoms.calc = calc
        atoms2.calc = self.calculator

        npt = Inhomogeneous_NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            pressure=pressure,
            compressibility=1e-5,
            taut=1e2 * fs,
            taup=1e2 * fs,
        )
        npt.run(nsteps)

        npt2 = Inhomogeneous_NPTBerendsen(
            atoms2,
            timestep=dt,
            temperature_K=temperature_K,
            pressure=pressure,
            compressibility=1e-5,
            taut=1e2 * fs,
            taup=1e2 * fs,
        )
        npt2.run(steps=nsteps)

        final_1 = npt.atoms.get_positions()
        final_2 = npt2.atoms.get_positions()

        np.testing.assert_allclose(final_1, final_2, atol=1e-8)
        np.testing.assert_allclose(
            npt.atoms.get_cell(), npt2.atoms.get_cell(), atol=1e-8
        )
