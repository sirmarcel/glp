import jax.numpy as jnp
from jax_md import space

from .utils import cast

def displacement_frac(cell):
    return space.periodic_general(cell, fractional_coordinates=True)[0]


def displacement_real(cell):
    return space.periodic_general(cell, fractional_coordinates=False)[0]


def to_frac(positions, cell):
    inverse = space.inverse(cell)
    return jnp.einsum("Aa,ia->iA", inverse, positions)


def from_frac(fractional, cell):
    return jnp.einsum("aA,iA->ia", cell, fractional)


def wrap(positions, cell):
    return from_frac(to_frac(positions, cell) % cast(1.0), cell)


def project_on(R, normals):
    return jnp.einsum("Ia,Aa->IA", R, normals)


def get_heights(cell, normals=None):
    if normals is None:
        normals = get_normals(cell)
    return jnp.diag(project_on(cell.T, normals))


def get_normals(cell):
    # surface normals of cell boundaries
    # (i.e. normalised lattice vectors of reciprocal lattice)
    # convention: indexed by the lattice vector they're not orthogonal to
    inv = space.inverse(cell)  # rows: inverse lattice vectors
    normals = inv / jnp.linalg.norm(inv, axis=1)[:, None]
    return normals


def project_on_normals(positions, cell):
    return project_on(positions, get_normals(cell))
