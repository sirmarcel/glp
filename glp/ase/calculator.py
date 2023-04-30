import jax
import numpy as np

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import compare_atoms, PropertyNotImplementedError
from ase.constraints import full_3x3_to_voigt_6_stress
from ase.units import fs

from glp import comms
from glp.system import atoms_to_system


class Calculator(GetPropertiesMixin):
    # ase/vibes compatibility. not used!
    name = "glp"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        get_calculator,
        atoms=None,
        dtype=jax.numpy.float32,  # note: this will only work if jax is imported with appropriate config
        raw=False,  # for debug only: do not do any post-processing
    ):
        self.calculate_fn, self.update_fn = None, None
        self.get_calculator = get_calculator
        self.dtype = dtype
        self.raw = raw

        self.atoms = None
        self.results = {}
        if atoms is not None:
            self.setup(atoms)

    @classmethod
    def instantiate(
        cls,
        potential,
        heat_flux=False,
        dtype=jax.numpy.float32,
        raw=False,
        **calculator_args,
    ):
        from glp import instantiate

        if isinstance(potential, dict):
            potential = instantiate.get_component("potential", potential)

        if not heat_flux:
            calculator = instantiate.get_component(
                "calculator", {"atom_pair": calculator_args}
            )

        else:
            if potential.cutoff == potential.effective_cutoff:
                calculator_args["heat_flux"] = True

                calculator = instantiate.get_component(
                    "calculator", {"atom_pair": calculator_args}
                )
            else:
                calculator = instantiate.get_component(
                    "calculator", {"heat_flux_unfolded": calculator_args}
                )

        return cls(lambda system: calculator(potential, system), dtype=dtype, raw=raw)

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)

        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()

            if self.need_setup(changes):
                self.setup(atoms)

    def need_setup(self, changes):
        return "pbc" in changes or "numbers" in changes

    def setup(self, atoms):
        system = atoms_to_system(atoms, dtype=self.dtype)
        calculator, state = self.get_calculator(system)
        self.calculate_fn = jax.jit(calculator.calculate)
        self.state = state
        self.atoms = atoms.copy()

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=None,
        **kwargs,
    ):
        self.update(atoms)

        system = atoms_to_system(atoms, dtype=self.dtype)
        velocities = jax.numpy.array(atoms.get_velocities(), dtype=self.dtype)
        masses = jax.numpy.array(atoms.get_masses(), dtype=self.dtype)

        results, state = self.calculate_fn(
            system, self.state, velocities=velocities, masses=masses
        )

        if state.overflow:
            comms.talk("overflow. redoing calculation...")
            self.setup(atoms)
            results, state = self.calculate_fn(
                system, self.state, velocities=velocities, masses=masses
            )
            if state.overflow:
                comms.warn(
                    "encountered immediate overflow; we cannot recover from this"
                )
                comms.warn(
                    "this can be caused by large cutoffs -- see preceding warnings!"
                )
                raise RuntimeError

        ase_results = {}
        if "energy" in results:
            ase_results["energy"] = np.array(results["energy"])

        if "forces" in results:
            ase_results["forces"] = np.array(results["forces"])

        if "stress" in results:
            if not self.raw:
                ase_results["stress"] = (
                    full_3x3_to_voigt_6_stress(np.array(results["stress"]))
                    / atoms.get_volume()
                )
            else:
                ase_results["stress"] = np.array(results["stress"])

        if "heat_flux" in results:
            for key in ["heat_flux", "heat_flux_convective", "heat_flux_potential"]:
                if key in results:
                    if not self.raw:
                        # apply vibes velocity convention to outputs!
                        ase_results[key] = (
                            1000.0 * fs * np.array(results[key]) / atoms.get_volume()
                        )
                    else:
                        ase_results[key] = np.array(results[key])

        self.state = state
        self.results = ase_results
        return ase_results

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(
                f"{name} property not present in results!"
            )

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None):
        return self.get_property(name="energy", atoms=atoms)
