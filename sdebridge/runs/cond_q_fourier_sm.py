import sys
from functools import partial
sys.path.append('../../')

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from matplotlib import colormaps

from sdebridge.sde import FourierGaussianKernelSDE
from sdebridge.diffusion_bridge import DiffusionBridge
from sdebridge.data_processing import butterfly_amasina_pts, butterfly_honrathi_pts
from sdebridge.utils import eval_score


def sample_ellipse(
    n_samples: int,
    scale: float = 1.0,
    shifts: jnp.ndarray = jnp.array([0.0, 0.0]),
    a: float = 1.0,
    b: float = 1.0,
) -> jnp.ndarray:
    theta = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)
    x = a * jnp.cos(theta)
    y = b * jnp.sin(theta)
    return scale * jnp.stack([x, y], axis=1) + shifts[None, :]

if __name__ == "__main__":
    n_samples = 64
    n_bases = 4

    S0 = sample_ellipse(n_samples, scale=0.5)
    print("S0 shape: ", S0.shape)

    sde_config = ConfigDict(
        {
            'init_S': S0,
            'n_bases': n_bases,
            'n_grid': 50,
            'grid_range': [-1.5, 1.5],
            'alpha': 0.1,
            'sigma': 0.2,
            'T': 1.0,
            'N': 50,
            'dim': 2
        }
    )

    sde = FourierGaussianKernelSDE(sde_config)
    bridge = DiffusionBridge(sde)

    X0 = jnp.zeros((sde.n_bases, 2), dtype=jnp.complex64)
    X0_flatten = jnp.concatenate((X0[:, 0], X0[:, 1]), axis=0)

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        for i in range(1):
            forward_uncond = bridge.simulate_forward_process(
                initial_val = X0_flatten,
                num_batches = 64
            )