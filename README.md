# Overview of dataset

We use nanoscale acoustic confinement to excite coherent acoustic phonons (dynamical strain) with near-terahertz frequency, and demonstrate that such phonons can easily be excited when breaking the film-substrate epitaxy in the metallic perovskite SrRuO3. Moreover, thermal annealing of the free-standing membranes placed on a substrate with a similar lattice constant dramatically decreases the interfacial acoustic impedance mismatch, evidencing the reformation of chemical bonds and interfacial restructuring.

## Installation
The analysis relies on several custom Python packages, including some [`polars`](https://pola.rs) utility packages (`polars-dataset`, `polars-complex`) and a Rust extension for `polars` (`polars-splines`), as well as styling utilities for `matplotlib` (`scientific_mplstyle`).

Setting up the `conda` environment can be done with [`make`](https://www.gnu.org/software/make/), as defined in the file `Makefile`, by running
```bash
make env
```

To activate the created environment, run
```bash
conda activate driven_nickelate
```

If `make` is not on your system or you prefer a manual install, see the instructions below.

### Manual installation

1. Create virtual environment with `conda` (from the root of the repository):
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate driven_nickelate
```

3. Compile `polars-splines` with `maturin`
```bash
maturin develop --release --manifest-path ./src/polars-splines/Cargo.toml
```

4. Install the remaining dependencies:
```bash
pip install ./src/scientific_mplstyle ./src/polars-dataset --no-build-isolation
```

5. Install the analysis package in editable mode:
```bash
pip install -e ./src/analysis --no-build-isolation
```

