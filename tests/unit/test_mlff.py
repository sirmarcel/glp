from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp

import numpy as np
from jax import jit

from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from glp import atoms_to_system, comms
from glp.instantiate import get_calculator
from glp.ase import Calculator
from glp.dynamics import atoms_to_input

n_samples = 3
skin = 0.1
skin_unfolder = 0.1

primitive = read("assets/snse/geometry.in.primitive", format="aims")


def calculate_numerical_stress(atoms, d=1e-6, voigt=True):
    """Calculate numerical stress using finite difference."""

    stress = np.zeros((3, 3), dtype=float)

    cell = atoms.cell.copy()
    V = atoms.get_volume()
    for i in range(3):
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = atoms.get_potential_energy()

        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = atoms.get_potential_energy()

        stress[i, i] = (eplus - eminus) / (2 * d * V)
        x[i, i] += d

        j = i - 2
        x[i, j] = d
        x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = atoms.get_potential_energy()

        x[i, j] = -d
        x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = atoms.get_potential_energy()

        stress[i, j] = (eplus - eminus) / (4 * d * V)
        stress[j, i] = stress[i, j]

    atoms.set_cell(cell, scale_atoms=True)

    if voigt:
        return stress.flat[[0, 4, 8, 5, 2, 1]]
    else:
        return stress


class TestMLFF(TestCase):
    def test_mega(self):
        x64 = True
        dtype = jnp.float64
        d_stress = 1e-4
        atol = 1e-4
        rtol = 1e-4

        initial_atoms = primitive * [4, 8, 8]
        initial_system = atoms_to_system(initial_atoms, dtype=dtype)

        def get_atoms(seed):
            atoms = initial_atoms.copy()
            atoms.rattle(stdev=0.01, seed=seed)

            rng = np.random.default_rng(seed)
            strain = 0.01 * rng.uniform(low=-1.0, high=1.0, size=(3, 3))
            strain = 0.5 * (strain + strain.T)
            strain += np.eye(3)

            cell = atoms.get_cell().array

            strained_cell = np.einsum("Ba,Aa->AB", strain, cell)

            atoms.set_cell(strained_cell, scale_atoms=True)

            MaxwellBoltzmannDistribution(
                atoms, temperature_K=10, rng=np.random.default_rng(seed)
            )

            return atoms

        samples = [get_atoms(i) for i in range(n_samples)]

        for m in [1, 2]:
            potential_config = {
                "mlff": {
                    "folder": f"assets/snse/model_m{m}/",
                    "dtype": dtype,
                }
            }

            calculator_configs = {
                # eq. 15
                "strain_system": {
                    "variations_end_to_end": {
                        "skin": skin,
                        "stress_mode": "strain_system",
                    }
                },
                # eq. 16
                "strain_graph": {
                    "variations_end_to_end": {
                        "skin": skin,
                        "stress_mode": "strain_graph",
                    }
                },
                # eq. 17
                "unfolded_strain_unfolded": {
                    "variations_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin_unfolder,
                        "stress_mode": "strain_unfolded",
                    }
                },
                # eq. 18
                "strain_direct": {
                    "variations_end_to_end": {
                        "skin": skin,
                        "stress_mode": "direct",
                    }
                },
                # eq. 19
                "atom_pair_direct": {
                    "atom_pair": {
                        "skin": skin,
                        "heat_flux": True,
                        "convective": False,
                    }
                },
                # eq. 20
                "unfolded_direct_unfolded": {
                    "variations_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin_unfolder,
                        "stress_mode": "direct_unfolded",
                    }
                },
                "hf_unfolded": {
                    "heat_flux_unfolded": {
                        "skin": skin,
                        "skin_unfolder": skin_unfolder,
                        "convective": False,
                    }
                },
                "hf_hardy": {
                    "heat_flux_hardy": {
                        "skin": skin,
                        "convective": False,
                    }
                },
            }

            calculators = {}

            for name, config in calculator_configs.items():
                calculators[name] = Calculator(
                    get_calculator(potential_config, config), dtype=dtype, raw=True
                )

            ase_calculator = Calculator(
                get_calculator(potential_config, {"atom_pair": {"skin": skin}}),
                dtype=dtype,
            )

            results = {}
            for name, calc in calculators.items():
                results[name] = [calc.calculate(atoms) for atoms in samples]

            reference_stress = []
            for atoms in samples:
                atoms.calc = ase_calculator
                reference_stress.append(
                    calculate_numerical_stress(atoms, voigt=False, d=d_stress)
                    * atoms.get_volume()
                )

            for key, res in results.items():
                if "stress" in res[0]:
                    print(f"stress: {key} @ M={m}")
                    for i in range(len(samples)):
                        np.testing.assert_allclose(
                            reference_stress[i], res[i]["stress"], atol=atol, rtol=rtol
                        )
                else:
                    print(f"skip {key}")

            print(f"hf @ M={m}")
            for i in range(len(samples)):
                np.testing.assert_allclose(
                    results["hf_hardy"][i]["heat_flux"],
                    results["hf_unfolded"][i]["heat_flux"],
                    atol=atol,
                    rtol=rtol,
                )
                if m == 1:
                    np.testing.assert_allclose(
                        results["hf_hardy"][i]["heat_flux"],
                        results["atom_pair_direct"][i]["heat_flux"],
                        atol=atol,
                        rtol=rtol,
                    )

            if x64:
                d = 1e-6

                def get_fd_forces(atoms, d=1e-6):
                    results = []
                    for i in [0, 10, 20]:
                        before = ase_calculator.calculate(atoms)["energy"]
                        this = atoms.copy()
                        this.positions[i, 0] += d
                        after = ase_calculator.calculate(this)["energy"]

                        f = (after - before) / d

                        results.append(-1.0 * f)

                    return np.array(results)

                fd_forces = [get_fd_forces(atoms, d=d) for atoms in samples]

                for key, res in results.items():
                    if "forces" in res[0]:
                        print(f"forces: {key} @ M={m}")
                        for i in range(len(samples)):
                            np.testing.assert_allclose(
                                res[i]["forces"][0, 0], fd_forces[i][0], atol=1e-5
                            )
                            np.testing.assert_allclose(
                                res[i]["forces"][10, 0], fd_forces[i][1], atol=1e-5
                            )
                            np.testing.assert_allclose(
                                res[i]["forces"][20, 0], fd_forces[i][2], atol=1e-5
                            )
                    else:
                        print(f"skip {key}")
