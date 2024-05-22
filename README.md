# SDE Bridges

We learn the score function for discretisations of infinite-dimensional stochastic differential equations.
We use this to bridge the SDEs conditional on an end set.

## Installation

To install the package, run the following command in the root directory of the repository (sdebridge):

```bash
pip install .
```


## Usage

The package provides a class SDE, for defining SDEs.
Definitions for the class and specific SDEs including Brownian motion and Kunita SDEs are provided in the module `sdebridge.sdes`.
In order to define reverse bridges, we can either define a score function or learn it using the module `sdebridge.diffusion_bridge`.

## Examples

Examples of how to use the package are provided in the `notebooks` directory.
