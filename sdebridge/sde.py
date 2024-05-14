import dataclasses
import functools
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from sdebridge.solver import euler_maruyama, gradients_and_covariances

# Kernels


def gaussian_Q_half_kernel(landmarks: jnp.ndarray, alpha: float, sigma: float) -> jnp.ndarray:
    xy_coords = landmarks.reshape(-1, 2)
    diff = xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :]
    dis = jnp.sum(jnp.square(diff), axis=-1)
    Q_half = alpha * jnp.exp(-dis / sigma**2)
    return Q_half


def gaussian_kernel_2d(alpha: float, sigma: float) -> callable:
    def k(x, y):
        return alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x - y), axis=-1) / (sigma**2))

    return k


def gaussian_kernel_independent(alpha, sigma, grid_range, grid_size, dim=2):
    grid = jnp.linspace(*grid_range, grid_size)
    grid = jnp.stack(jnp.meshgrid(grid, grid, indexing="xy"), axis=-1)
    grid_ = grid.reshape(-1, dim)

    gauss_kernel = gaussian_kernel_2d(alpha, sigma)
    batch_over_grid = jax.vmap(gauss_kernel, in_axes=(None, 0))
    batch_over_vals = jax.vmap(batch_over_grid, in_axes=(0, None))

    def kernel(val):
        Q_half = batch_over_vals(val, grid_)
        Q_half = Q_half.reshape(-1, grid_size**2)
        return Q_half

    return kernel


@dataclasses.dataclass(frozen=True, eq=True)
class SDE:
    T: float
    N: int
    dim: int
    n_bases: int
    drift: Callable
    diffusion: Callable
    bm_shape: tuple
    params: Any

    @property
    def dt(self):
        return self.T / self.N

    @property
    def ts(self):
        return jnp.linspace(0, self.T, self.N)

    @partial(jax.jit, static_argnums=0)
    def covariance(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """covariance term: sigma @ sigma^T"""
        _diffusion = self.diffusion(val, time)
        return jnp.matmul(_diffusion, jnp.conj(_diffusion).T)

    @partial(jax.jit, static_argnums=0)
    def div_covariance(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """divergence of covariance term: nabla_x (sigma @ sigma^T)"""
        _jacobian = jax.jacfwd(self.covariance, argnums=0)(val, time)
        return jnp.trace(_jacobian)

    @partial(jax.jit, static_argnums=(0, 2))
    def simulate_trajectories(self, initial_val: jnp.ndarray, num_batches: int, key: jax.Array):
        keys = jax.random.split(key, num_batches)
        euler = functools.partial(
            euler_maruyama,
            x0=initial_val,
            ts=self.ts,
            drift=self.drift,
            diffusion=self.diffusion,
            bm_shape=self.bm_shape,
        )
        batched_eul = jax.vmap(euler, in_axes=0)
        trajectories = batched_eul(keys)
        return trajectories

    # @partial(jax.jit, static_argnums=0)
    def grad_and_covariance(self, trajs):
        gradients, covariances = gradients_and_covariances(trajs, self.ts, self.drift, self.diffusion)
        return gradients, covariances


def reverse(sde, score_fun):
    """Drift and diffusion for backward bridge process (Z*(t)):
        dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        score_fun (callable): nabla log p(x, t), either a closed form or a neural network.

    Returns:
        results: trajectories: jax.Array, (B, N, d) backward bridge trajectories
    !!! N.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
    """

    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        inverted_time = sde.T - time
        rev_drift_term = sde.drift(val=val, time=inverted_time)

        _score = jnp.squeeze(score_fun(val=val, time=inverted_time))
        _covariance = sde.covariance(val=val, time=inverted_time)
        score_term = jnp.dot(_covariance, _score)

        div_term = sde.div_covariance(val=val, time=inverted_time)
        return -rev_drift_term + score_term + div_term

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        inverted_time = sde.T - time
        return sde.diffusion(val=val, time=inverted_time)

    return SDE(
        T=sde.T,
        N=sde.N,
        dim=sde.dim,
        n_bases=sde.n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=sde.bm_shape,
        params=sde.params,
    )


def bridge(sde, score_fun):
    """Drift and diffusion for the forward bridge process (X*(t)) which is the "backward of backward":
        dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

    Args:
        score_fun  (callable): nabla log h(x, t), either a closed form or a neural network.

    Returns:
        results: "trajectories": jax.Array, (B, N, d) forward bridge trajectories (in normal order)
    """

    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        orig_drift_term = sde.drift(val=val, time=time)

        _score = jnp.squeeze(score_fun(val=val, time=time))
        _covariance = sde.covariance(val=val, time=time)
        score_term = jnp.dot(_covariance, _score)
        return orig_drift_term + score_term

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return sde.diffusion(val=val, time=time)

    return SDE(
        T=sde.T,
        N=sde.N,
        dim=sde.dim,
        n_bases=sde.n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=sde.bm_shape,
        params=sde.params,
    )


def brownian_sde(T, N, dim, n_bases, sigma):
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 2
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 2
        return sigma * jnp.eye(n_bases)

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(n_bases, dim),
        params=None,
    )


def trace_brownian_sde(T, N, dim, n_bases, alpha, power):
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 2
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 2
        k = jnp.arange(1, n_bases + 1)
        return jnp.diag(alpha / k**power)

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(n_bases, dim),
        params=None,
    )


def gaussian_kernel_sde(T, N, dim, n_bases, alpha, sigma):
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return gaussian_Q_half_kernel(landmarks=val, alpha=alpha, sigma=sigma)

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(n_bases, dim),
        params=None,
    )


def gaussian_independent_kernel_sde(
    T: float,
    N: int,
    dim: int,
    n_bases: int,
    alpha: float,
    sigma: float,
    grid_range: tuple,
    grid_size: int,
):
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 2
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 2
        kernel = gaussian_kernel_independent(alpha, sigma, grid_range=grid_range, grid_size=grid_size)
        val = val.reshape(-1, dim)
        Q_half = kernel(val)
        # Q_half = jnp.kron(Q_half, jnp.eye(2))
        return Q_half

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(grid_size**2, dim),
        params=None,
    )


def fourier_gaussian_kernel_sde(T, N, dim, n_bases, alpha, sigma, n_grid, grid_range, n_samples):
    if n_samples / 2 + 1 < n_bases:
        raise ValueError("(n_samples/2 + 1)  must be more than n_bases")

    def inverse_fourier(coefficients, num_pts):
        """Array of shape [..., 2*num_bases, dim]
        Returns array of shape [..., num_pts, dim]"""
        assert coefficients.shape[-2] % 2 == 0
        num_bases = int(coefficients.shape[-2] / 2)
        coeffs_real = coefficients[..., :num_bases, :]
        coeffs_im = coefficients[..., num_bases:, :]
        complex_coefficients = coeffs_real + 1j * coeffs_im
        return jnp.fft.irfft(complex_coefficients, norm="forward", n=num_pts, axis=-2)

    gaussian_grid_kernel = gaussian_kernel_independent(alpha, sigma, grid_range, n_grid, dim)

    def evaluate_Q(X_coeffs: jnp.ndarray) -> jnp.ndarray:
        X_pts = inverse_fourier(X_coeffs, n_samples)  # (n_bases, 2)
        Q_half = gaussian_grid_kernel(X_pts)
        return Q_half.reshape(n_samples, n_grid, n_grid)

    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    @jax.jit
    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        Q_eval = evaluate_Q(val)
        Q_eval = jnp.fft.rfft(Q_eval, axis=0, norm="forward", n=2 * (n_bases - 1))
        Q_eval = jnp.fft.ifft2(Q_eval, axes=(1, 2), norm="backward")

        diff = Q_eval.reshape(n_bases, n_grid**2)
        coeffs = jnp.stack([diff.real, diff.imag], axis=0)
        coeffs = coeffs.reshape(*diff.shape[:-2], -1, diff.shape[-1])

        return coeffs

    return SDE(T, N, dim, 2 * n_bases, drift, diffusion, bm_shape=(n_grid**2, dim), params=None)
