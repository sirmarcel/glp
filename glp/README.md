## Technical Notes

### Running the tests

The unit tests require `pytest`, `mlff`, and `ase`. The integration tests also require `fhi-vibes`.

Then:

```
cd tests/unit
# if no GPU is present, it is best to run
pytest --ignore test_mlff.py
# (the MLFF test is *very* expensive on CPU)

# otherwise, you can just go with
pytest

cd ../integration/vibes
sh run.sh

cd ../mlff
sh run.py
```


### Practices

- We cast all numerical literals appearing in the code to `jnp.array` via `glp.utils.cast`. Allegedly this avoids re-compilations. This applies to parameters (cutoffs, LJ parameters, etc.) as well.
- Wherever there is any question of `dtype`, we defer to input `dtype`. We avoid casting main inputs into other types. Therefore, `glp` should work in double precision.

### Todos

- [ ] Custom JVP to make derivatives of m.i.c. displacements correct
- [ ] Make jit-table update of unfolding, following the pattern of the nighborlist
- [ ] Upgrade neighborlist to cell list (enable scaling to larger systems)
