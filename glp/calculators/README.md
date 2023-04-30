## calculators

This implements different ways of turning a graph potential into a calculator, i.e., something that can actually compute energy, forces, stress, and heat flux.

### Terminology

`atom_pair` indicates that the "atom-pair" vectors `R_i - R_j`, the edges in the graph, are treated as fundamental. Derivatives are computed with respect to edges, and forces and stress (and if applicable, heat flux) are computed accordingly. `end_to_end`, on the other hand, indicates that the positions in the simulation cell and the basis (or lattice) vectors are treated as fundamental. `unfolded`, finally, uses the "unfolded" simulation cell as fundamental. All perspectives are ultimately equivalent, but are treated explicitly here to allow for verification of this statement.

For the stress, we generally distinguish between `direct` approaches, where derivatives with respect to inputs are taken directly and used to compute the stress, and `strain` approaches where a strain transformation is explicitly applied to inputs. Different ways of computing both are compared in the `variations_` calculators. These variations arise from the ability to use the graph OR the positions+basis OR the unfolded system as equivalant inputs. Each can be strained, or used directly. (One additional difficulty arises from the fact that in the standard minimum image convention, the derivatives with respect to the cell are not as expected, and so straining the positions+basis is not always correct, unless an adapted mic is used.)

### Recommendations for Production

For energy, forces, and stress, `atom_pair.py` or `end_to_end.py` should be used. Both are virtually identical.

For heat flux, for `M=1` (only nearest-neighbor interactions), `atom_pair.py` should be used, as it is very fast. Otherwise, `heat_flux_unfolded.py` must be used, which is slightly slower, but still scales linearly with system size.

In the special case of cells which are smaller than the cutoff radius, the `supercell.py` calculator can be used. Heat flux is not implemented in this case, as the GK method typically requires supercells anyhow.

### Variations for Testing

Many other variations are implemented for testing and comparison.

- `heat_flux_hardy.py` implements the "Hardy" heat flux, using the minimum image convention. It scales *quadratically* with system size and is therefore *not* recommended.
- `variations_*.py` contain various different ways of computing the stress, with and without explicit strain transformations. They can be safely ignored, unless implementation details are being investigated.
