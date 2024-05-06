from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from .sde import SDE


def batch_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Batch matrix multiplication"""
    return jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)(A, B)


@partial(jax.jit, static_argnums=(0,))
def euler_maruyama(
    sde: SDE,
    initial_vals: jnp.ndarray,
    terminal_vals: jnp.ndarray,
    rng_key: jax.Array,
) -> dict:
    """Euler-Maruyama solver for SDEs

    initial_vals: (B, 2*N), complex64
    terminal_vals: (B, 2*N), complex64
    """
    enforce_terminal_constraint = terminal_vals is not None

    SolverState = namedtuple("SolverState", ["vals", "grads", "covs", "step_key"])
    init_state = SolverState(
        vals=initial_vals,
        grads=jnp.empty_like(initial_vals),
        covs=jnp.empty(
            (initial_vals.shape[0], sde.dim * sde.n_bases, sde.dim * sde.n_bases),
            dtype=jnp.complex64,
        ),
        step_key=rng_key,
    )

    @partial(jax.jit, static_argnums=(0, 1))
    def euler_maruyama_step(state: SolverState, time: jnp.ndarray) -> tuple:
        """Euler-Maruyama step, NOTE: all the calculations are over batches"""
        time = jnp.expand_dims(time, axis=-1)
        time = jnp.tile(time, (state.vals.shape[0], 1))
        step_key, _ = jax.random.split(state.step_key)
        _drift = jax.vmap(sde.drift, in_axes=(0, 0))(state.vals, time)  # (b, 2*n_bases)
        drift_step = _drift * sde.dt

        n_batches = state.vals.shape[0]
        _brownian = jax.random.normal(
            step_key, shape=(n_batches, sde.dim * sde.n_grid**2)
        )  # (B, 2*n_grid**2)
        brownian_step = _brownian * jnp.sqrt(sde.dt)

        _diffusion = jax.vmap(sde.diffusion, in_axes=(0, None))(
            state.vals, time
        )  # (B, 2*n_bases, 2*n_grid**2)
        diffusion_step = batch_matmul(_diffusion, brownian_step)  # (B, 2*n_bases)

        _covariance = jax.vmap(sde.covariance, in_axes=(0, None))(
            state.vals, time
        )  # (B, 2*n_bases, 2*n_bases)
        _inv_covariance = jax.vmap(
            partial(jnp.linalg.pinv, hermitian=True, rcond=None)
        )(
            _covariance
        )  # (B, 2*n_bases, 2*n_bases)

        grads = (
            -batch_matmul(_inv_covariance, diffusion_step) / sde.dt
        )  # (B, 2*n_bases)

        new_vals = state.vals + drift_step + diffusion_step  # (B, 2*n_bases)
        new_state = SolverState(
            vals=new_vals,
            grads=grads,
            covs=_covariance,
            step_key=step_key,
        )
        return new_state, (
            state.vals,
            state.grads,
            state.covs,
            state.step_key,
        )

    _, (trajectories, gradients, covariances, step_keys) = jax.lax.scan(
        euler_maruyama_step,
        init=init_state,
        xs=(sde.ts[:-1]),
        length=sde.N,
    )

    if enforce_terminal_constraint:
        trajectories = trajectories.at[-1].set(terminal_vals)
    return {
        "trajectories": jnp.swapaxes(trajectories, 0, 1),
        "gradients": jnp.swapaxes(gradients, 0, 1),
        "covariances": jnp.swapaxes(covariances, 0, 1),
        "last_key": step_keys[-1],
    }
