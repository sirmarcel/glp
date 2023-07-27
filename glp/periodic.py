"""tools for dealing with periodicity

Conventions:
- `a,b,c,...` indicate real-space cartesian directions
- `A,B,C,...` indicate lattice vectors or inverse lattice vectors
- `R` are real-space vectors
- `X` are fractional-coordinate vectors

"""

from jax import vmap
import jax.numpy as jnp
from functools import partial

from .utils import cast


def inverse(cell):
    return jnp.linalg.inv(cell)


def _to_frac(cell, R):
    return jnp.einsum("Aa,a->A", inverse(cell), R)


def to_frac(cell, R):
    return vmap(partial(_to_frac, cell))(R)


def _from_frac(cell, X):
    return jnp.einsum("aA,A->a", cell, X)


def from_frac(cell, X):
    return vmap(partial(_from_frac, cell))(X)


def make_displacement(cell):
    return partial(displacement, cell)


def displacement(cell, Ra, Rb):
    if cell is None:
        return Rb - Ra

    else:
        R = Rb - Ra
        X = _to_frac(cell, R)
        X = jnp.mod(X + cast(0.5), cast(1.0)) - cast(0.5)

        return _from_frac(cell, X)


def wrap(cell, R):
    return from_frac(cell, to_frac(cell, R) % cast(1.0))


def project_on(normals, R):
    return jnp.einsum("Aa,ia->iA", normals, R)


def get_heights(cell):
    normals = get_normals(cell)
    return jnp.diag(project_on(normals, cell.T))


def get_normals(cell):
    # surface normals of cell boundaries
    # (i.e. normalised lattice vectors of reciprocal lattice)
    # convention: indexed by the lattice vector they're not orthogonal to
    inv = inverse(cell)  # rows: inverse lattice vectors
    normals = inv / jnp.linalg.norm(inv, axis=1)[:, None]
    return normals


def project_on_normals(cell, R):
    return project_on(get_normals(cell), R)
