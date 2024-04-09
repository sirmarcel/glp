from unittest import TestCase

import numpy as np
import jax.numpy as jnp
import jax
from time import monotonic

from ase.build import bulk


from glp.neighborlist import quadratic_neighbor_list, neighbor_list, check_reallocation_cell_list
from glp.system import atoms_to_system, unfold_system
from glp.graph import system_to_graph
from glp.unfold import unfolder


def random_movements(n, amount):
    directions = np.random.random((n, 3))
    directions /= np.sqrt(np.sum(directions**2, axis=1))[:, None]

    magnitudes = np.random.random(n) * amount

    return directions * magnitudes[:, None]


def compare_distances(a, b, cutoff, atol=0.0, rtol=1e-7):
    a = filter_and_sort(a, cutoff)
    b = filter_and_sort(b, cutoff)

    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def filter_and_sort(distances, cutoff):
    d = distances[distances < cutoff]
    d = d[d > 0]
    return np.sort(d)


def get_distances(graph):
    from glp.utils import distance

    return distance(graph.edges)[graph.mask]


class TestNeighborList(TestCase):
    # Checking cell list
    def test_basic(self):
        cutoff = 5.0
        skin = 0.5
        atoms = bulk("Ar", cubic=False) * [7, 7, 7]

        system = atoms_to_system(atoms)

        allocate, update, need_update = quadratic_neighbor_list(
            system.cell, cutoff, skin, debug=True
        )
        neighbors = allocate(system.R)
        neighbors = update(system.R, neighbors)
        distances = get_distances(system_to_graph(system, neighbors))

        compare_distances(
            atoms.get_all_distances(mic=True), distances, cutoff, atol=5e-6
        )

        atoms.positions += random_movements(len(atoms), 0.5 * skin)
        system = atoms_to_system(atoms)

        assert not need_update(neighbors, system.R, system.cell)

        atoms.positions += random_movements(len(atoms), 0.1 * skin)
        system = atoms_to_system(atoms)

        assert need_update(neighbors, system.R, system.cell)

        neighbors = update(system.R, neighbors)
        distances = get_distances(system_to_graph(system, neighbors))

        compare_distances(
            atoms.get_all_distances(mic=True), distances, cutoff, atol=5e-6
        )

    def test_hard(self):
        from ase import Atoms
        from ase.cell import Cell

        from ase.neighborlist import neighbor_list as ase_nl

        cutoff = 4.25
        skin = 0.0
        scaled_positions = np.random.random((600, 3))

        def get_sizes(idx, n):
            idx = jnp.array(idx, dtype=int)
            return jax.ops.segment_sum(jnp.ones_like(idx), idx, n)

        for cell in [
            Cell.new(np.array([9, 9, 9, 90, 90, 90])),
            # Cell.new(np.array([12, 20, 11, 60, 100, 70])),
        ]:
            # note the *critical* role of pbc=True here (lol)
            atoms = Atoms(cell=cell, scaled_positions=scaled_positions, pbc=True)

            system = atoms_to_system(atoms)

            neighbors, update = neighbor_list(system, cutoff=cutoff, skin=skin)
            assert not neighbors.overflow

            graph = system_to_graph(system, neighbors)
            my_distances = get_distances(graph)

            idx_i, idx_j, ase_distances = ase_nl("ijd", atoms, cutoff)
            ase_sizes = get_sizes(idx_i, len(atoms))

            my_sizes = get_sizes(neighbors.centers, len(atoms) + 1)

            offset = 0
            for i, size in enumerate(ase_sizes):
                assert my_sizes[i] == size
                ase_idx = idx_j[offset : offset + size]
                my_idx = graph.others[offset : offset + size]

                np.testing.assert_array_equal(np.sort(ase_idx), np.sort(my_idx))

                offset += size

            # now we make sure that we catch when the cell becomes too small!

            atoms.set_cell(atoms.get_cell() * 0.8)
            system = atoms_to_system(atoms)

            neighbors = update(atoms_to_system(atoms), neighbors)
            assert neighbors.overflow

    def test_variable_cell(self):
        cutoff = 5.0
        skin = 0.5
        spread = 0.5
        atoms = bulk("Ar", cubic=False) * [7, 7, 7]
        system = atoms_to_system(atoms)

        allocate, update, need_update = quadratic_neighbor_list(
            system.cell, cutoff, skin, debug=True, capacity_multiplier=1.5, use_cell_list=True
        )
        neighbors = allocate(system.R)
        neighbors = update(system.R, neighbors)
        distances = get_distances(system_to_graph(system, neighbors))

        compare_distances(
            atoms.get_all_distances(mic=True), distances, cutoff, atol=5e-6
        )

        cell = atoms.get_cell().array
        cell = cell @ (
            np.eye(3) + (spread * np.random.default_rng(1).random((3, 3)) - spread / 2)
        )
        atoms.set_cell(cell, scale_atoms=True)

        system = atoms_to_system(atoms)

        assert need_update(neighbors, system.R, system.cell)

        neighbors = update(system.R, neighbors, new_cell=system.cell)

        assert not neighbors.overflow

        distances = get_distances(system_to_graph(system, neighbors))

        compare_distances(
            atoms.get_all_distances(mic=True), distances, cutoff, atol=5e-6
        )

    def test_jit_high_level_interface(self):
        cutoff = 5.0
        skin = 0.5
        atoms = bulk("Ar", cubic=True) * [7, 7, 7]

        system = atoms_to_system(atoms)

        neighbors, update = neighbor_list(system, cutoff=cutoff, skin=skin)
        update = jax.jit(update)

        start = monotonic()
        neighbors = update(system, neighbors)
        assert (
            monotonic() - start > 0.25
        )  # this is stupid -- if your computer is fast this fails

        distances = get_distances(system_to_graph(system, neighbors))

        compare_distances(
            atoms.get_all_distances(mic=True), distances, cutoff, atol=1e-5
        )

        atoms.positions += random_movements(len(atoms), 0.6 * skin)
        system = atoms_to_system(atoms)

        start = monotonic()
        neighbors = update(system, neighbors)
        assert monotonic() - start < 0.1

        distances = get_distances(system_to_graph(system, neighbors))

        compare_distances(
            atoms.get_all_distances(mic=True), distances, cutoff, atol=1e-5
        )

    def test_unfolding(self):
        from glp.utils import distance

        cutoff = 5.0
        skin = 0.5
        atoms = bulk("Ar", cubic=True) * [7, 7, 7]

        system = atoms_to_system(atoms)

        unfolding, update_unfolding = unfolder(system, cutoff, skin)

        big = unfold_system(system, unfolding)

        neighbors, update = neighbor_list(big, cutoff=cutoff, skin=skin)

        distances = distance(system_to_graph(big, neighbors).edges)
        ase_distances = atoms.get_all_distances(mic=True)

        for i in range(len(atoms)):
            compare_distances(
                ase_distances[i], distances[neighbors.centers == i], cutoff, atol=1e-5
            )

        atoms.positions += random_movements(len(atoms), 0.6 * skin)
        system = atoms_to_system(atoms)

        unfolding = update_unfolding(system, unfolding)
        big = unfold_system(system, unfolding)
        neighbors = update(big, neighbors)

        distances = distance(system_to_graph(big, neighbors).edges)
        ase_distances = atoms.get_all_distances(mic=True)

        for i in range(len(atoms)):
            compare_distances(
                ase_distances[i], distances[neighbors.centers == i], cutoff, atol=1e-5
            )