from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from .sde import SDE


def batch_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Batch matrix multiplication"""
    return jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)(A, B)


def euler_maruyama(key, x0, ts, drift, diffusion, bm_shape=None):
    if bm_shape is None:
        bm_shape = x0.shape

    def step_fun(key_and_x_and_t, dt):
        k, x, t = key_and_x_and_t
        k, subkey = jax.random.split(k, num=2)
        eps = jax.random.normal(subkey, shape=bm_shape)
        diffusion_ = diffusion(x, t)
        xnew = x + dt * drift(x, t) + jnp.sqrt(dt) * diffusion_ @ eps
        tnew = t + dt

        return (k, xnew, tnew), xnew

    init = (key, x0, ts[0])
    _, x_all = jax.lax.scan(step_fun, xs=jnp.diff(ts), init=init)
    return jnp.concatenate([x0[None], x_all], axis=0)


def gradients_and_covariances(xs, ts, drift, diffusion):
    @jax.jit
    def grad_and_cov(t0: float, X0: jax.Array, t1: float, X1: jax.Array):
        dt = t1 - t0
        drift_last = drift(X0, t0)
        diffusion_last = diffusion(X0, t0)
        cov = diffusion_last @ diffusion_last.T
        inv_cov = invert(diffusion_last, diffusion_last.T)
        grad = 1 / dt * inv_cov @ (X1 - X0 - dt * drift_last)
        return -grad, cov

    grad_cov_fn = jax.vmap(grad_and_cov, in_axes=(0, 0, 0, 0))
    mult_trajectories = jax.vmap(grad_cov_fn, in_axes=(None, 0, None, 0))
    return mult_trajectories(ts[:-1], xs[:, :-1], ts[1:], xs[:, 1:])


def invert(mat, mat_transpose):
    """
    Inversion of mat*mat_transpose.
    :param mat: array of shape (n, m) i.e. ndim=2
    :param mat_transpose: array with shape (m, n) with ndim=2
    :return: (mat*mat.T)^{-1} with shape (n, n)
    """
    return jnp.linalg.inv(mat @ mat_transpose)


# @partial(jax.jit, static_argnums=(0,))
def euler_maruyama2(
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
        ),
        step_key=rng_key,
    )

    # @partial(jax.jit, static_argnums=(0, 1))
    def euler_maruyama_step(state: SolverState, time: jnp.ndarray) -> tuple:
        """Euler-Maruyama step, NOTE: all the calculations are over batches"""
        time = jnp.expand_dims(time, axis=-1)
        time = jnp.tile(time, (state.vals.shape[0], 1))
        step_key, _ = jax.random.split(state.step_key)
        _drift = jax.vmap(sde.drift, in_axes=(0, 0))(state.vals, time)  # (b, 2*n_bases)
        drift_step = _drift * sde.dt

        n_batches = state.vals.shape[0]
        _brownian = jax.random.normal(
            step_key, shape=(n_batches, sde.bm_shape)
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

        _inv_covariance = jax.vmap(
            partial(jnp.linalg.pinv, hermitian=False, rcond=None)
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
        xs=(sde.ts),
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
