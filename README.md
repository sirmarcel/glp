# `glp`
## calculators for graph-based machine-learning potentials

`glp` turns any graph-based machine-learning potential implemented in `jax` into a `jit`-able calculator ready for molecular dynamics. It provides unified implementations of forces, stress, and (with caveats) the heat flux.

In a nutshell: If you provide a function that maps a `Graph` of atom-pair vectors to atomic potential `energies`, `glp` takes care of everything else. ✨

With `glp`, we aim to make it straightforward to go from implementing a new machine-learning potential to using it for moderate-sized production runs (thousands of atoms, nanoseconds of MD). `glp` also serves as an illustration of implementations of stress with automatic differentiation. It was written to support the following [manuscript](https://arxiv.org/abs/2305.01401):

```
Stress and heat flux via automatic differentiation
by Marcel F. Langer, J. Thorben Frank, and Florian Knoop

arXiv:2305.01401
```

If you use this code in an academic context, please cite this preprint. If you use the heat flux, please also cite [this other preprint](https://arxiv.org/abs/2303.14434). 🙏

**`glp` is a fairly early-stage code. There is not stable release cycle yet!**

See also: [`gkx`](https://github.com/sirmarcel/gkx), which implements the remaining parts for a fully `jax`-based Green-Kubo workflow.

## Features

Once you supply a `Potential` with the required interface (see below), `glp` gives you:

- `ase` `Calculator` with `FHI-vibes` integration that can compute energy, forces, stress, and heat flux
- `jit`-ready calculation functions as building blocks for more advanced workflows for energy, forces, stress and heat flux
- Full support for periodic systems, including those with non-orthorhombic cells, and including any changes of those cells
- Velocity verlet integrator ready for `lax.scan` (mostly for fun, but fast nevertheless)

To our knowledge, it is the first tool that implements the heat flux, required for the calculation of thermal conductivity with the Green-Kubo method, for autodiff-based machine-learning potentials.

If you don't want to implement a potential, we also have a direct interface with [`mlff`](http://github.com/thorben-frank/mlff) that implements the state-of-the-art `so3krates` MLP.

### Developer Features™

Apart from these more "user-facing" features, `glp` also

- takes care of all the logic required to pad and invalidate neighborlists, even if the simulation cell changes.
- provides a simple interface for taking care of `jit`-ing dynamics simulations where inputs can change.
- demonstrates how to implement forces, stress, and heat flux in different ways with automatic differentiation.
- is extensively tested for correctness of the implementation of these quantities.

### Caveats

`glp` currently can't do the following:

- Support training of MLPs (even though it could be pressed into service with some hacking)
- Support truly non-local MLPs (forces will likely work, but don't really require the whole machinery of `glp`)
- Scale to very large systems, because the neighbourlist is implemented naively and scales quadratically with system size. It's fine for serious runs with up to a few thousand atoms.
- Take care of full MD workflows: we don't do the simulation (except, experimentally, NVE) and we don't take care of the surrounding logic of logging, i/o, etc.

## Interface

### `Potential`

A `Potential` is a function that maps a `Graph` to an `ndarray` of atomic potential energies.

It is declared in `glp` as:

```python
from glp import Potential

potential = Potential(potential_fn, cutoff)

```

For forces and stress, this represents *all* `glp` needs to know about your potential. For heat flux, you additionally need to supply an `effective_cutoff` that quantifies the maximum range of interactions beyond local environments, for instance through message-passing (where it is `n_interactions * cutoff`).

A `Potential` currently can't carry internal state, and is not expected to have non-local interactions; we basically expect either a potential that works within neighbourhoods or with message passing.

### `Graph`

The input data structure for a `Potential` is

```python
Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others", "mask"))
```

We adopt the "sparse" convention of `jax-md`, where we store flat `ndarrays` of pairs `n_pairs` rather than the more old-school "dense" format where we store `n_atoms x n_neighbors`.

Then:

- `edges` are atom-pair vectors `r_ij = r_j - r_i` (with the minimum image convention applied)
- `nodes` are atomic numbers (may be used for different node labels in the future)
- `centers` are indices indicating which `i` belongs to which entry in `edges`
- `others` are corresponding indices `j`
- `mask` is a boolean array indicating which entries in `edges` are padding (indices for these are set to `n_atoms`)

For an example, you may want to check the [`lennard_jones`](https://github.com/sirmarcel/glp/blob/main/glp/potentials/lj.py) potential. 

### `calculators` and `state`

The real heart of `glp` are the `calculators`, which are the functions that actually make use of a `Potential` to compute quantities of interest. Due to the constraints of `jax`, they are a bit more complicated than one would maybe expect: They have to balance the need for being `jit`-able, which needs static input/output shapes, with the fact that the system can change unpredictably during simulation.

Luckily, we already did all of the work and you can just use the ready-made calculators! Nevertheless, to use them, you need to understand how they work a bit.

Our solution to this `jit` problem is to carry around a `state`, which is some `namedtuple` (or more generally any `pytree`) that carries around all the runtime information needed between steps. It can be updated within the `jit` at every step, and it can yield an *overflow* when changes have accumulated that can't be dealt with in a `jit`-able way. If an overflow occurs, we basically create the calculator again and get a new `state` to start from. Overflows are indicated by `state.overflow == True`. **If this occurs, the results returned by the calculator are wrong and must be discarded!** (For those familiar with `jax-md`, this is essentially how their `NeighborList` operates -- we just extend this slightly to support other transformations.)

A `Calculator` is therefore defined as *two* functions:

```python
Calculator = namedtuple("Calculator", ("calculate", "update"))
```

`calculate` is a `jit`-able function from `System, state` (`System` is defined below) to `results, state`. `update` takes only a `System` and returns a new `Calculator` and a new `state`.

The `calculator` functions implemented in `glp` currently (for an overview see the readme in `glp/calculators`) all have the following interface:

```python
from glp.calculators.some_calculator import calculator

my_calculator = calculator(
	system, # example of the system we want to investigate
	potential, # instance of Potential
	skin=0.1, # additional cutoff used for caching the neighborlist
	capacity_multiplier=1.25, # add 25% more pairs to neighborlist as spare capacity
	**kwargs, # additional arguments
	)
```

It's often inconvenient to keep track of all the arguments, so in some places, a `get_calculator` function is used that is simply a `partial` of `calculator` that fixes all arguments except `system`. The `.update` function is basically the same thing: it takes a `system` and returns a `Calulator, state` tuple.

An overview of available calculators is given in [here](https://github.com/sirmarcel/glp/tree/main/glp/calculators). In general, we recommend:

- [`atom_pair`](https://github.com/sirmarcel/glp/blob/main/glp/calculators/atom_pair.py) for energy, forces, stress (equivalent to [`end_to_end`](https://github.com/sirmarcel/glp/blob/main/glp/calculators/end_to_end.py))
- [`atom_pair`](https://github.com/sirmarcel/glp/blob/main/glp/calculators/atom_pair.py) with `heat_flux=True` for heat flux for models without message passing. It will produce *wrong* results for models with message passing.
- [`heat_flux_unfolded`](https://github.com/sirmarcel/glp/blob/main/glp/calculators/heat_flux_unfolded.py) for heat flux for models *with* message passing. This calculator constructs an extended "unfolded" simulation cell (see [this preprint](https://arxiv.org/abs/2303.14434)) and therefore is slower and needs more memory than others.
- [`supercell`](https://github.com/sirmarcel/glp/blob/main/glp/calculators/supercell.py) for energy, forces, stress in the case where simulation cells are smaller than the cutoff of the potential (for instance, relaxing primitive cells)

Other calculators exist to illustrate different approaches but are not typically needed for production use.

### System

The `System` is the internal representation of an arrangement of atoms:

```python

System = namedtuple("System", ("R", "Z", "cell"))
```

Where `R` are the positions, `Z` atomic numbers, and `cell` either lattice vectors (**each vector a column**) or `None` (for molecules).

You will likely not have to instantiate this by hand, `glp` provides an `atoms_to_system()` method that turns an `ase.Atoms` into a `System`.


### `instantiate`: So3krates or Lennard-Jones

If you already have a [`mlff`](http://github.com/thorben-frank/mlff) model or want to test things with a Lennard-Jones potential, you can make use of the `glp.instantiate` module, which implements a convenience interface to create the various things defined by `glp` from [`specable`](https://github.com/sirmarcel/specable)-style dictionaries.

It works like this:

```python
from glp import instantiate

potential_dict = {"mlff": {"folder": "path/to/model"}}
potential_dict = {"lennard_jones": {"sigma": 3.405, "epsilon": 0.01042, "cutoff": 10.5, "onset": 9.0}}

# get a get_calculator function with a given potential and calculator
get_calculator = instantiate.get_calculator(potential_dict, {"atom_pair": {"skin": 0.1}})

# ... just for jax:
calculator, state = get_calculator(system)

# ... or for ase/vibes:
from glp.ase import Calculator

calculator = Calculator(get_calculator)

```

For `FHI-vibes`, you can also use:

```
[calculator]
name:                          calculator
module:                        glp.vibes

[calculator.parameters.calculator]
calculator:                     atom_pair
skin:                           0.1

[calculator.parameters.potential]
potential:                      lennard_jones
sigma:                          3.405
# ...
```

## Installation

First, you need to make sure to have a working [`jax`](https://github.com/google/jax/) installation. Then:


```
git clone git@github.com:sirmarcel/glp.git
cd glp

# minimal install
pip install .

```

The `ase`, `mlff`, and `vibes` interfaces require these libraries to be installed. 


For `ase` and `vibes`, you can let `pip` do this for you:

```

# for everything (vibes, ase)
pip install ".[full]"

# or...
pip install ".[vibes]"
pip install ".[ase]"

```

Currently, for some reason, [`mlff`](https://github.com/thorben-frank/mlff) must be installed manually by running

```
git clone git@github.com:thorben-frank/mlff.git
cd mlff
pip install -e .
```

## Units and Conventions

`glp` is essentially unit agnostic, it simply manipulates numbers as given. The `dynamics` module assumes everything is in `ase` units, so the timestep is in `ase.units.fs`. For compatibility with `ase`, it is best to stick to Ångstrom for distances and eV for energies.

Stress and heat flux are internally computed *without* dividing by the volume, but the standard convention is enforced in the `ase` calculator, where we also convert the heat flux to being in SI-ps instead of `ase`-fs, in line with the `FHI-vibes` conventions. To circumenvent all such processing, the `raw=True` argument can be passed.

## What about `jax-md`?

Much of the original design of `glp` was heavily inspired by `jax-md`. Since then, we've decided to take some slightly different technical directions, which allow us to take a more direct approach to treating periodic systems and the associated tasks of computing the stress.

In `jax-md`, potentials are typically defined via their parameters and a `displacement_fn` (computing minimum image convention atom-pair vectors) that encodes the simulation `cell`. This means that the `cell` is a *parameter*, rather than a direct *argument* of the potential energy function, so taking derivatives with respect to the cell or computing the stress is a bit awkward (but possible). It also means that the potential is responsible for mapping the `displacement_fn` over pairs of neighbours, typically with the help of a `neighborlist`. At the time of writing, `jax-md` also implements the displacement in general periodic systems such that the derivative with respect to the cell is incorrect, so the stress must be computed by transforming the displacements.

In contrast, `glp` takes the perspective that potentials are strictly functions mapping `(positions, charges, cell)` to a potential energy. It provides tools to easily implement a particular class of such general potentials: Those that internally represent `(positions, charges, cell)` as a graph of minimum-image convention atom-pair vectors between nodes (atoms) labeled with charges. We take care of constructing this graph and therefore take charge of dealing with displacement functions and neighborlists, which are no longer the responsibility of a given potential.

This means that the function signatures involved cleanly represent that underlying concepts, and so we don't need to jump through any hoops to obtain the stress with AD -- with whatever implementation we'd like, even those that directly transform positions and cell.

In any case, it should be fairly straightforward to adopt `jax_md.energy` functions for `glp`, if desired, and we are looking forward to PRs in that direction!

## Want to know more?

Please also see the technical documentation of `glp` in the `glp` folder. Otherwise, just read the code! It's probably about the same number of lines as this readme. ☺️

## What's next?

For `glp`, the next steps are outlined in the technical readme: there is some work to be done to scale to larger systems and have less catastrophic reactions to an overflow of the unfolding procedure which is needed for the heat flux.

In a larger context, `glp` aims to be a neat building block for larger systems. In particular, we're planning to use it as one part of a future molecular dynamics package, the seeds of which you can find in `glp/dynamics` and `mlff/mdx`. However, we're still gathering some experience on how these tools work in practice, and are thinking about the correct abstractions to build this thing.

[The best is yet to come!](https://www.youtube.com/watch?v=B-Jq26BCwDs)
