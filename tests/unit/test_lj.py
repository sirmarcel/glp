from unittest import TestCase

from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np
from jax import jit

from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.stress import voigt_6_to_full_3x3_stress
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from glp.instantiate import get_calculator
from glp.ase import Calculator

n_samples = 3
skin = 1.0


def get_atoms(seed):
    atoms = bulk("Ar", cubic=False) * [8, 9, 10]
    atoms.rattle(stdev=0.01, seed=seed)

    rng = np.random.default_rng(seed)
    strain = 0.05 * rng.uniform(low=-1.0, high=1.0, size=(3, 3))
    strain = 0.5 * (strain + strain.T)
    strain += np.eye(3)

    cell = atoms.get_cell().array

    strained_cell = np.einsum("Ba,Aa->AB", strain, cell)

    atoms.set_cell(strained_cell, scale_atoms=True)

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=10, rng=np.random.default_rng(seed)
    )

    return atoms


def get_heat_flux(atoms, convective=True):
    stresses = (
        voigt_6_to_full_3x3_stress(atoms.calc.results["stresses"])
        * atoms.get_volume()
        * -1
    )
    heat_flux_potential = np.einsum("iab,ib", stresses, atoms.get_velocities())
    energies = atoms.calc.results["energies"] + 0.5 * np.sum(
        atoms.get_velocities() * atoms.get_momenta(), axis=1
    )
    heat_flux_convective = np.einsum("i,ia->a", energies, atoms.get_velocities())

    if convective:
        heat_flux = heat_flux_potential + heat_flux_convective

        return {
            "heat_flux": heat_flux,
            "heat_flux_potential": heat_flux_potential,
            "heat_flux_convective": heat_flux_convective,
        }

    else:
        heat_flux = heat_flux_potential

        return {
            "heat_flux": heat_flux,
            "heat_flux_potential": heat_flux_potential,
        }


class TestCalculatorWithLJ(TestCase):
    def test_mega(self):
        samples = [get_atoms(i) for i in range(n_samples)]
        reference_calculator = LennardJones(
            epsilon=0.01042, sigma=3.405, rc=10.5, ro=9.0, smooth=True
        )
        reference_results = []
        for atoms in samples:
            atoms.calc = reference_calculator
            atoms.get_potential_energy()
            results = atoms.calc.results
            results["stress"] = atoms.get_stress(voigt=False) * atoms.get_volume()
            hf = get_heat_flux(atoms)
            for k, v in hf.items():
                results[k] = v

            reference_results.append(results)

        for x64 in [True, False]:
            if x64:
                dtype = jnp.float64
                atol = 5e-5
                rtol = 1e-6
            else:
                dtype = jnp.float32
                atol = 1e-3
                rtol = 1e-4

            potential_config = {
                "lennard_jones": {
                    "sigma": 3.405,
                    "epsilon": 0.01042,
                    "cutoff": 10.5,
                    "onset": 9.0,
                }
            }

            calculator_configs = [
                {"atom_pair": {"convective": True, "skin": skin, "heat_flux": True}},
                {"end_to_end": {"skin": skin}},
                {"supercell": {"skin": skin, "n_replicas": 1}},
                {"variations_atom_pair": {"skin": skin, "fractional_mic": True}},
                {"variations_atom_pair": {"skin": skin, "fractional_mic": False}},
                {
                    "variations_end_to_end": {
                        "skin": skin,
                        "stress_mode": "strain_system",
                    }
                },
                {
                    "variations_end_to_end": {
                        "skin": skin,
                        "stress_mode": "strain_graph",
                    }
                },
                {"variations_end_to_end": {"skin": skin, "stress_mode": "direct"}},
                {
                    "variations_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin,
                        "stress_mode": "direct_system",
                    }
                },
                {
                    "variations_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin,
                        "stress_mode": "direct_unfolded",
                    }
                },
                {
                    "variations_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin,
                        "stress_mode": "strain_unfolded",
                    }
                },
                {
                    "heat_flux_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin,
                        "convective": True,
                    }
                },
                {"heat_flux_hardy": {"skin": skin, "convective": True}},
            ]

            for config in calculator_configs:
                calculator = Calculator(
                    get_calculator(potential_config, config), dtype=dtype, raw=True
                )

                for i, atoms in enumerate(samples):
                    reference = reference_results[i]
                    results = calculator.calculate(atoms)

                    for key in results.keys():
                        # print(f"{i}, {k}")
                        np.testing.assert_allclose(
                            results[key], reference[key], rtol=rtol, atol=atol
                        )
