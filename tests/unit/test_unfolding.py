from unittest import TestCase

import numpy as np

# np.random.seed(1)
from ase import Atoms

from glp.unfold import unfolder, unfold
from glp.system import atoms_to_system, unfold_system


def make_unfolded_atoms(system, unfolding):
    unfolded = unfold_system(system, unfolding)

    return Atoms(positions=unfolded.R[np.where(unfolded.padding_mask)], pbc=False)


def compare_distances(a, b, cutoff, atol=0.0, rtol=1e-7):
    a = filter_and_sort(a, cutoff)
    b = filter_and_sort(b, cutoff)

    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def filter_and_sort(distances, cutoff):
    d = distances[distances < cutoff]
    d = d[d > 0]
    return np.sort(d)


def compare_all_distances(atoms, unfolded, cutoff, atol=0.0, rtol=1e-7):
    dist = atoms.get_all_distances(mic=True)
    unfolded_dist = unfolded.get_all_distances(mic=False)

    for i in range(len(atoms)):
        try:
            compare_distances(dist[i], unfolded_dist[i], cutoff, atol=atol, rtol=rtol)
        except AssertionError:
            a = dist[i]
            b = unfolded_dist[i]

            a = filter_and_sort(a, cutoff)
            b = filter_and_sort(b, cutoff)

            print(f"idx {i}: {a.shape}, {b.shape}")
            raise


def compare_all_distances_with_nl(atoms, unfolded, cutoff, atol=0.0, rtol=1e-7):
    """Slower, but treats multiple replicas"""
    from ase.neighborlist import neighbor_list

    dist, idx_dist = neighbor_list("di", atoms, cutoff, self_interaction=False)
    unfolded_dist, idx_unfolded_dist = neighbor_list(
        "di", unfolded, cutoff, self_interaction=False
    )

    for i in range(len(atoms)):
        a = dist[np.where(idx_dist == i)]
        b = unfolded_dist[np.where(idx_unfolded_dist == i)]
        try:
            compare_distances(a, b, cutoff, atol=atol, rtol=rtol)
        except AssertionError:
            print(f"idx {i}: {a.shape}, {b.shape}")
            raise


def perturb_along_normal(atoms, spread, basis=0):
    import numpy as np
    from collections import namedtuple

    # basis: the original cell object
    # normals: the normal vectors of the unit cell surfaces (pointing inwards)
    #          since each surface is spanned by two basis vectors, only one points
    #          out of that surface. normals are indexed such that they match.
    # heights: the distance of the basis vector pointing out of each surface
    Cell = namedtuple("Cell", ("basis", "normals", "heights"))

    def get_cell(cell):
        from ase.cell import Cell as aseCell

        cell = aseCell(cell)
        normals = get_normal_vectors(cell)
        heights = get_heights(cell, normals)

        return Cell(cell, normals, heights)

    def get_normal_vectors(cell):
        reciprocal = cell.reciprocal()
        normals = reciprocal / np.linalg.norm(reciprocal, axis=1)[:, None]
        return normals

    def get_heights(cell, normals):
        return np.diag(cell @ normals.T)

    atoms = atoms.copy()

    cell = get_cell(atoms.get_cell())
    direction = cell.normals[basis]
    positions = atoms.get_positions()

    randoms = np.random.random(len(atoms)) * 2 * spread - spread
    positions += randoms[:, None] * direction

    atoms.set_positions(positions)

    return atoms


def perturb_random(atoms, spread):
    atoms = atoms.copy()
    atoms.positions += np.random.random((len(atoms), 3)) * spread - spread / 2

    return atoms


class TestUnfolder(TestCase):
    def test_distances(self):
        for cutoff, cell in [
            (0.29, np.array([[5.1, 0.0, 0.0], [-2.5, 4.42, 0.0], [0.0, 0.0, 13.0]])),
            (0.39, np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])),
        ]:
            n = 250
            skin = 0.1
            spread = 0.1

            # generate positions that are slightly outside the u.c.
            # (this case is very common in MD, and we need to treat
            # it in a reasonable way!)
            positions = np.random.random((n, 3)) * (1 + 2 * spread) - spread

            # test: does this work at all? do we get the same distances?
            atoms = Atoms(scaled_positions=positions, cell=cell, pbc=True)

            system = atoms_to_system(atoms)

            unfolding, update, need_update = unfolder(system, cutoff, skin, debug=True)
            unfolded_atoms = make_unfolded_atoms(system, unfolding)

            compare_all_distances(atoms, unfolded_atoms, cutoff, atol=1e-5)

            # test: move atoms w/o triggering recomputation
            # case 1: tiny random changes
            atoms2 = perturb_random(atoms, 0.05)
            system2 = atoms_to_system(atoms2)
            self.assertFalse(need_update(system2, unfolding))

            # case 2: change orthogonal to surfaces
            for i in range(3):
                atoms2 = perturb_along_normal(atoms, skin / 2, basis=i)
                system2 = atoms_to_system(atoms2)
                self.assertFalse(need_update(system2, unfolding))

            unfolding = update(system2, unfolding, force_update=False)

            assert not unfolding.overflow
            unfolded_atoms = make_unfolded_atoms(system2, unfolding)
            compare_all_distances(atoms2, unfolded_atoms, cutoff, atol=1e-5)

            # test: move atoms *further*, triggering recomputation
            atoms3 = perturb_along_normal(atoms2, skin, basis=0)
            atoms3 = perturb_along_normal(atoms3, skin, basis=1)
            system3 = atoms_to_system(atoms3)
            self.assertTrue(need_update(system3, unfolding))

            unfolding = update(system3, unfolding)
            assert not unfolding.overflow

            unfolded_atoms = make_unfolded_atoms(system3, unfolding)
            compare_all_distances(atoms3, unfolded_atoms, cutoff, atol=1e-5)

    def test_large_cutoff(self):
        n = 10
        skin = 0.1
        spread = 0.1
        cutoff = 0.89
        cell = np.eye(3)

        positions = np.random.random((n, 3)) * (1 + 2 * spread) - spread

        atoms = Atoms(scaled_positions=positions, cell=cell, pbc=True)
        system = atoms_to_system(atoms)

        unfolding, need_update = unfolder(system, cutoff, skin)

        unfolded_atoms = make_unfolded_atoms(system, unfolding)

        compare_all_distances_with_nl(atoms, unfolded_atoms, cutoff, atol=1e-6)
