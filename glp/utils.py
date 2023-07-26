from jax import jit, vmap
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnums=2)
def boolean_mask_1d(masked, mask, out_size, filler):
    # implements: masked[mask] = out in a jittable way.
    # result: out[0:mask.sum()] containing all entries
    # where mask == True, and the rest padded with filler
    # note that the last entry of out will always be filler

    # print("boolean_mask_1d")  # uncomment for jit testing
    out = jnp.ones(out_size, dtype=masked.dtype) * filler

    cumsum = jnp.cumsum(mask)
    locations = jnp.where(mask, cumsum - 1, out_size - 1)
    out = out.at[locations].set(jnp.where(mask, masked, filler))

    n_matches = cumsum[-1]
    overflow = n_matches + 1 > out_size

    return out, overflow


def cast(x):
    """Cast number literal to jnp.ndarray.

    This avoids jit recompiles, as native python types
    are "weak" types in jax. This makes everything explicit.
    In high-precision situations, jax type promotion should™
    do the right thing.
    """

    if type(x) == int:
        return jnp.array(x, dtype=jnp.int32)
    elif type(x) == float:
        return jnp.array(x, dtype=jnp.float32)
    else:
        raise ValueError(f"cannot cast {x} of as type {type(x)} is unknown to me")


def str_to_dtype(string):
    if not isinstance(string, str):
        return string

    if string == "float32":
        return jnp.float32
    elif string == "float64":
        return jnp.float64
    else:
        raise ValueError(f"unknown dtype {string}")

def squared_distance(R):
    return jnp.sum(R ** cast(2.0), axis=-1)


def distance(R):
    r2 = squared_distance(R)
    mask = r2 > cast(0)
    safe_r2 = jnp.where(mask, r2, cast(0))
    return jnp.where(mask, jnp.sqrt(safe_r2), cast(0))
