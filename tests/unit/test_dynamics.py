from unittest import TestCase

import numpy as np

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax.lax import scan


from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs
from ase.calculators.lj import LennardJones
from ase.units import fs
from ase.md import VelocityVerlet

from glp.ase.calculator import Calculator

from glp.system import atoms_to_system
from glp.potentials import lennard_jones
from glp.calculators import atom_pair

from glp.dynamics import verlet as verlet_dynamics
from glp.dynamics import atoms_to_input

class TestLJ(TestCase):
    def setUp(self):
        self.sigma = 2.0
        self.epsilon = 1.5
        self.rc = 6.0
        self.ro = 5.0

        self.calculator = LennardJones(
            epsilon=self.epsilon, sigma=self.sigma, rc=self.rc, ro=self.ro, smooth=True
        )

        self.potential = lennard_jones(
            epsilon=self.epsilon, sigma=self.sigma, cutoff=self.rc, onset=self.ro
        )

    def test(self):
        dt = 1 * fs
        nsteps = 100

        atoms = bulk("Ar", cubic=False) * [5, 5, 5]
        MaxwellBoltzmannDistribution(atoms, temperature_K=10)
        Stationary(atoms)
        atoms2 = atoms.copy()
        atoms3 = atoms.copy()

        system = atoms_to_system(atoms3, dtype=jnp.float64)
        masses = jnp.array(atoms3.get_masses(), dtype=jnp.float64)
        velocities = jnp.array(atoms3.get_velocities(), dtype=jnp.float64)

        get_calculator = lambda system: atom_pair.calculator(
            self.potential, system, skin=0.1, heat_flux=False
        )

        calc = Calculator(get_calculator, atoms, dtype=jnp.float64)

        atoms.calc = calc
        atoms2.calc = self.calculator

        verlet = VelocityVerlet(atoms, timestep=dt)
        verlet.run(steps=nsteps)

        verlet2 = VelocityVerlet(atoms2, timestep=dt)
        verlet2.run(steps=nsteps)

        system, velocities, masses = atoms_to_input(atoms3, dtype=jnp.float64)
        
        dynamics, state = verlet_dynamics(
            system, velocities, masses, get_calculator, dt
        )
        state, steps = scan(dynamics.step, state, None, length=nsteps)
        points, results, overflows = steps

        final_1 = verlet.atoms.get_positions()
        final_2 = verlet2.atoms.get_positions()
        final_3 = np.array(points.R[-1])

        np.testing.assert_allclose(final_1, final_2, atol=1e-8)
        np.testing.assert_allclose(final_1, final_3, atol=1e-8)
