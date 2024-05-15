import dataclasses
import functools
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp


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
def cov(sde: SDE, val: jnp.ndarray, time: jnp.ndarray) -> jax.Array:
    """covariance term: sigma @ sigma^T"""
    _diffusion = sde.diffusion(val, time)
    return jnp.matmul(_diffusion, _diffusion.T)


@partial(jax.jit, static_argnums=0)
def cov_div(sde: SDE, val: jnp.ndarray, time: jnp.ndarray) -> jax.Array:
    """divergence of covariance term: nabla_x (sigma @ sigma^T)"""
    jacobian_ = jax.jacfwd(cov, argnums=1)(sde, val, time)
    return jnp.trace(jacobian_)


@partial(jax.jit, static_argnums=(0, 2))
def simulate_traj(sde: SDE, initial_val: jax.Array, num_batches: int, key: jax.Array):
    keys = jax.random.split(key, num_batches)
    euler = functools.partial(
        euler_maruyama,
        x0=initial_val,
        ts=sde.ts,
        drift=sde.drift,
        diffusion=sde.diffusion,
        bm_shape=sde.bm_shape,
    )
    batched_eul = jax.vmap(euler, in_axes=0)
    trajectories = batched_eul(keys)
    return trajectories


@partial(jax.jit, static_argnums=(0, 2))
def simulate_traj_grad_cov(sde: SDE, initial_val, num_batches, key):
    keys = jax.random.split(key, num_batches)
    egc = functools.partial(euler_maruyama_grad_cov, x0=initial_val, sde=sde)
    traj, grad, cov = jax.vmap(egc)(keys)
    return traj, grad, cov


def reverse(sde: SDE, score_fun: Callable) -> SDE:
    """Drift and diffusion for backward bridge process (Z*(t)):
        dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        score_fun (callable): nabla log p(x, t), either a closed form or a neural network.

    Returns:
        results: trajectories: jax.Array, (B, N, d) backward bridge trajectories
    !!! N.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
    """

    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        inverted_time = sde.T - time
        rev_drift_term = sde.drift(val=val, time=inverted_time)

        _score = jnp.squeeze(score_fun(val=val, time=inverted_time))
        _covariance = cov(sde, val=val, time=inverted_time)
        score_term = jnp.dot(_covariance, _score)

        div_term = cov_div(sde, val=val, time=inverted_time)
        return -rev_drift_term + score_term + div_term

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
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


def bridge(sde: SDE, score_fun: Callable) -> SDE:
    """Drift and diffusion for the forward bridge process (X*(t)) which is the "backward of backward":
        dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

    Args:
        score_fun  (callable): nabla log h(x, t), either a closed form or a neural network.

    Returns:
        results: "trajectories": jax.Array, (B, N, d) forward bridge trajectories (in normal order)
    """

    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        orig_drift_term = sde.drift(val=val, time=time)

        _score = jnp.squeeze(score_fun(val=val, time=time))
        _covariance = sde.cov(val=val, time=time)
        score_term = jnp.dot(_covariance, _score)
        return orig_drift_term + score_term

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
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


@partial(jax.jit, static_argnums=2)
def euler_maruyama_grad_cov(
    key,
    x0,
    sde,
) -> tuple:
    def step_fun(key_x_t, dt):
        k, x, t = key_x_t
        k, subkey = jax.random.split(k, num=2)
        eps = jax.random.normal(subkey, shape=sde.bm_shape)
        diffusion = sde.diffusion(x, t) @ eps

        xnew = x + dt * sde.drift(x, t) + jnp.sqrt(dt) * diffusion
        tnew = t + dt
        covnew = cov(sde, x, t)
        gradnew = -1 / dt * jnp.linalg.inv(covnew) @ diffusion

        return (k, xnew, tnew), (xnew, gradnew, covnew)

    init = (key, x0, sde.ts[0])
    _, (xs, grads, covs) = jax.lax.scan(step_fun, xs=jnp.diff(sde.ts), init=init)

    return xs, grads, covs


def brownian_sde(T, N, dim, n_bases, sigma) -> SDE:
    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        assert val.ndim == 2
        return jnp.zeros_like(val)

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
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


def trace_brownian_sde(T, N, dim, n_bases, alpha, power) -> SDE:
    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        assert val.ndim == 2
        return jnp.zeros_like(val)

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
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


def gaussian_kernel_sde(T, N, dim, n_bases, alpha, sigma) -> SDE:
    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.zeros_like(val)

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
        return kernel_gaussian_Q_half(landmarks=val, alpha=alpha, sigma=sigma)

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
) -> SDE:
    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        assert val.ndim == 2
        return jnp.zeros_like(val)

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
        assert val.ndim == 2
        kernel = kernel_gaussian_independent(alpha, sigma, grid_range=grid_range, grid_size=grid_size)
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


def fourier_gaussian_kernel_sde(T, N, dim, n_bases, alpha, sigma, n_grid, grid_range, n_samples) -> SDE:
    # if n_samples / 2 + 1 < n_bases:
    #     raise ValueError("(n_samples/2 + 1)  must be more than n_bases")

    def inverse_fourier(coefficients, num_pts):
        """Array of shape [..., 2*num_bases, dim]
        Returns array of shape [..., num_pts, dim]"""
        assert coefficients.shape[-2] % 2 == 0
        num_bases = int(coefficients.shape[-2] / 2)
        coeffs_real = coefficients[..., :num_bases, :]
        coeffs_im = coefficients[..., num_bases:, :]
        complex_coefficients = coeffs_real + 1j * coeffs_im
        return jnp.fft.irfft(complex_coefficients, norm="forward", n=num_pts, axis=-2)

    gaussian_grid_kernel = kernel_gaussian_independent(alpha, sigma, grid_range, n_grid, dim)

    def evaluate_Q(X_coeffs: jax.Array) -> jax.Array:
        X_pts = inverse_fourier(X_coeffs, n_samples)  # (n_bases, 2)
        Q_half = gaussian_grid_kernel(X_pts)
        return Q_half.reshape(n_samples, n_grid, n_grid)

    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        return jnp.zeros_like(val)

    @jax.jit
    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
        Q_eval = evaluate_Q(val)
        Q_eval = jnp.fft.rfft(Q_eval, axis=0, norm="forward", n=2 * (n_bases - 1))
        Q_eval = jnp.fft.ifft2(Q_eval, axes=(1, 2), norm="backward")

        diff = Q_eval.reshape(n_bases, n_grid**2)
        coeffs = jnp.stack([diff.real, diff.imag], axis=0)
        coeffs = coeffs.reshape(*diff.shape[:-2], -1, diff.shape[-1])

        return coeffs

    return SDE(T, N, dim, 2 * n_bases, drift, diffusion, bm_shape=(n_grid**2, dim), params=None)


def kernel_gaussian_Q_half(landmarks: jnp.ndarray, alpha: float, sigma: float) -> jax.Array:
    xy_coords = landmarks.reshape(-1, 2)
    diff = xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :]
    dis = jnp.sum(jnp.square(diff), axis=-1)
    Q_half = alpha * jnp.exp(-dis / sigma**2)
    return Q_half


def kernel_gaussian_2d(alpha: float, sigma: float) -> callable:
    def k(x, y):
        return alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x - y), axis=-1) / (sigma**2))

    return k


def kernel_gaussian_independent(alpha, sigma, grid_range, grid_size, dim=2):
    grid = jnp.linspace(*grid_range, grid_size)
    grid = jnp.stack(jnp.meshgrid(grid, grid, indexing="xy"), axis=-1)
    grid_ = grid.reshape(-1, dim)

    gauss_kernel = kernel_gaussian_2d(alpha, sigma)
    batch_over_grid = jax.vmap(gauss_kernel, in_axes=(None, 0))
    batch_over_vals = jax.vmap(batch_over_grid, in_axes=(0, None))

    def kernel(val):
        Q_half = batch_over_vals(val, grid_)
        Q_half = Q_half.reshape(-1, grid_size**2)
        return Q_half

    return kernel
