import dataclasses
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from einops import rearrange

from sdebridge.solver import euler_maruyama, gradients_and_covariances

# Kernels


def gaussian_Q_half_kernel(landmarks: jnp.ndarray, alpha: float, sigma: float) -> jnp.ndarray:
    xy_coords = landmarks.reshape(-1, 2)
    diff = xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :]
    dis = jnp.sum(jnp.square(diff), axis=-1)
    kernel = alpha * jnp.exp(-dis / sigma**2)
    Q_half = jnp.kron(kernel, jnp.eye(2))
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


@dataclasses.dataclass
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

    def covariance(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """covariance term: sigma @ sigma^T"""
        _diffusion = self.diffusion(val, time)
        return jnp.matmul(_diffusion, jnp.conj(_diffusion).T)

    def div_covariance(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """divergence of covariance term: nabla_x (sigma @ sigma^T)"""
        _jacobian = jax.jacfwd(self.covariance, argnums=0)(val, time)
        # _jacobian = jax.jacfwd(self.covariance, argnums=0, holomorphic=True)(val, time)
        # return jnp.trace(_jacobian, axis1=-2, axis2=-1)
        return jnp.trace(_jacobian)

    def simulate_trajectories(self, initial_val: jnp.ndarray, num_batches: int, key: jax.Array):
        keys = jax.random.split(key, num_batches)
        batched = jax.vmap(euler_maruyama, in_axes=(0, None, None, None, None, None))
        trajectories = batched(
            keys,
            initial_val,
            self.ts,
            self.drift,
            self.diffusion,
            self.bm_shape,
        )
        return trajectories

    def grad_and_covariance(self, trajs):
        gradients, covariances = gradients_and_covariances(
            trajs, self.ts, self.drift, self.diffusion
        )
        return gradients, covariances


def reverse(sde, score_fun):
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
    """Bridge the SDE using either pre-assigned score h or neural network approximation"""

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
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        assert val.ndim == 1
        return sigma * jnp.eye(val.size)

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(dim * n_bases,),
        params=None,
    )


def damped_brownian_sde(T, N, dim, n_bases, alpha):
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return alpha * jnp.diag(
            1.0 / jnp.concatenate([jnp.arange(1, dim // 2 + 1), jnp.arange(1, dim // 2 + 1)])
        )

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(dim * n_bases,),
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
        bm_shape=(dim * n_bases,),
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
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        kernel = gaussian_kernel_independent(
            alpha, sigma, grid_range=grid_range, grid_size=grid_size
        )
        val = val.reshape(-1, dim)
        Q_half = kernel(val)
        Q_half = jnp.kron(Q_half, jnp.eye(2))
        return Q_half

    return SDE(
        T=T,
        N=N,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_shape=(grid_size**2 * dim,),
        params=None,
    )


def fourier_gaussian_kernel_sde(
    T, N, dim, n_bases, alpha, sigma, init_S, n_grid, grid_range, n_samples
):
    n_padding = (n_samples - n_bases) // 2

    base_fn = lambda freq: jnp.exp(1j * jnp.arange(-n_samples // 2, n_samples // 2) * freq)
    freqs = jnp.fft.fftshift(jnp.fft.fftfreq(n_bases, d=1 / (2.0 * jnp.pi)))
    fourier_basis = jax.vmap(base_fn)(freqs)  # (n_bases, n_samples)

    grid = jnp.linspace(grid_range[0], grid_range[1], n_grid)
    grid = jnp.stack(jnp.meshgrid(grid, grid, indexing="xy"), axis=-1)  # (n_grid, n_grid, 2)

    def evaluate_S(S: jnp.ndarray) -> jnp.ndarray:
        S_coeffs = jnp.fft.fft(S, n=n_samples, axis=0) / jnp.sqrt(n_samples)
        S_coeffs = jnp.fft.fftshift(S_coeffs, axes=0)  # (n_samples, 2)
        S_eval = jnp.matmul(fourier_basis, S_coeffs) / jnp.sqrt(n_samples)  # (n_bases, 2)
        return S_eval

    def evaluate_X(val: jnp.ndarray) -> jnp.ndarray:
        X_coeffs = jnp.stack(jnp.split(val, 2, axis=-1), axis=-1)  # (n_bases, 2)
        X_coeffs = X_coeffs / jnp.sqrt(n_samples)
        X_coeffs = jnp.pad(
            X_coeffs,
            ((n_padding, n_padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )  # (n_samples, 2)
        X_eval = jnp.matmul(fourier_basis, X_coeffs) / jnp.sqrt(n_samples)  # (n_bases, 2)
        return X_eval

    def evaluate_Q(val: jnp.ndarray) -> jnp.ndarray:
        kernel = gaussian_kernel_2d(alpha, sigma)
        X_eval = evaluate_X(val)
        S_eval = X_eval + evaluate_S(init_S)  # (n_bases, 2)
        Q_eval = jax.vmap(
            jax.vmap(
                jax.vmap(
                    kernel,
                    in_axes=(None, 0),
                    out_axes=0,
                ),
                in_axes=(None, 1),
                out_axes=1,
            ),
            in_axes=(0, None),
            out_axes=0,
        )(
            S_eval, grid
        )  # (n_bases, n_grid, n_grid)
        return Q_eval

    def evaluate_diffusion(val: jnp.ndarray) -> jnp.ndarray:
        Q_eval = evaluate_Q(val)
        diffusion = jnp.fft.fft2(Q_eval, axes=(1, 2), norm="ortho")  # (n_bases, n_grid, n_grid)
        diffusion = jnp.fft.ifft(diffusion, axis=0, norm="ortho")  # (n_bases, n_grid, n_grid
        diffusion = rearrange(diffusion, "b g1 g2 -> b (g1 g2)")  # (n_bases, n_grid**2)
        return diffusion

    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        diffusion_ = evaluate_diffusion(val)
        return jnp.kron(jnp.eye(2), diffusion_)

    return SDE(T, N, dim, n_bases, drift, diffusion, bm_shape=(grid.size,), params=None)


# class FourierGaussianKernelSDE(SDE):
#     def __init__(self, config):
#         super().__init__(config)
#         self._fourier_basis = self.fourier_basis()
#         self._grid = self.grid()
#         self._init_S_eval = self.evaluate_S(self.init_S)
#
#     @property
#     def alpha(self) -> float:
#         return self.config.alpha
#
#     @property
#     def sigma(self) -> float:
#         return self.config.sigma
#
#     @property
#     def init_S(self) -> jnp.ndarray:
#         return self.config.init_S
#
#     @property
#     def n_samples(self) -> int:
#         return self.config.init_S.shape[0]
#
#     @property
#     def n_bases(self) -> int:
#         return self.config.n_bases
#
#     @property
#     def n_padding(self) -> int:
#         return (self.n_samples - self.n_bases) // 2
#
#     @property
#     def n_grid(self) -> int:
#         return self.config.n_grid
#
#     # @property
#     def fourier_basis(self) -> jnp.ndarray:
#         base_fn = lambda freq: jnp.exp(1j * jnp.arange(-self.n_samples // 2, self.n_samples // 2) * freq)
#         freqs = jnp.fft.fftshift(
#             jnp.fft.fftfreq(self.n_bases, d=1 / (2.0 * jnp.pi))
#         )
#         return jax.vmap(base_fn)(freqs)  # (n_bases, n_samples)
#
#     # @property
#     def grid(self) -> jnp.ndarray:
#         grid_range = self.config.grid_range
#         grid = jnp.linspace(grid_range[0], grid_range[1], self.n_grid)
#         grid = jnp.stack(jnp.meshgrid(grid, grid, indexing='xy'), axis=-1)
#         return grid  # (n_grid, n_grid, 2)
#
#     @partial(jax.jit, static_argnums=(0,))
#     def evaluate_S(self, S: jnp.ndarray) -> jnp.ndarray:
#         S_coeffs = jnp.fft.fft(S, n=self.n_samples, axis=0) / jnp.sqrt(self.n_samples)
#         S_coeffs = jnp.fft.fftshift(S_coeffs, axes=0)  # (n_samples, 2)
#         S_eval = jnp.matmul(self._fourier_basis, S_coeffs) / jnp.sqrt(self.n_samples)  # (n_bases, 2)
#         return S_eval
#
#     @partial(jax.jit, static_argnums=(0,))
#     def evaluate_X(self, val: jnp.ndarray) -> jnp.ndarray:
#         X_coeffs = jnp.stack(jnp.split(val, 2, axis=-1), axis=-1)  # (n_bases, 2)
#         X_coeffs = X_coeffs / jnp.sqrt(self.n_samples)
#         X_coeffs = jnp.pad(X_coeffs, ((self.n_padding, self.n_padding), (0, 0)), mode='constant',
#                            constant_values=0)  # (n_samples, 2)
#         X_eval = jnp.matmul(self._fourier_basis, X_coeffs) / jnp.sqrt(self.n_samples)  # (n_bases, 2)
#         return X_eval
#
#     @partial(jax.jit, static_argnums=(0,))
#     def evaluate_Q(self, val: jnp.ndarray) -> jnp.ndarray:
#         X_eval = self.evaluate_X(val)
#         S_eval = X_eval + self._init_S_eval  # (n_bases, 2)
#         Q_eval = jax.vmap(
#             jax.vmap(
#                 jax.vmap(
#                     partial(gaussian_kernel_2d, alpha=self.alpha, sigma=self.sigma),
#                     in_axes=(None, 0),
#                     out_axes=0
#                 ),
#                 in_axes=(None, 1),
#                 out_axes=1
#             ),
#             in_axes=(0, None),
#             out_axes=0
#         )(S_eval, self._grid)  # (n_bases, n_grid, n_grid)
#         return Q_eval
#
#     @partial(jax.jit, static_argnums=(0,))
#     def evaluate_diffusion(self, val: jnp.ndarray) -> jnp.ndarray:
#         Q_eval = self.evaluate_Q(val)
#         diffusion = jnp.fft.fft2(Q_eval, axes=(1, 2), norm='ortho')  # (n_bases, n_grid, n_grid)
#         diffusion = jnp.fft.ifft(diffusion, axis=0, norm='ortho')  # (n_bases, n_grid, n_grid
#         diffusion = rearrange(diffusion, 'b g1 g2 -> b (g1 g2)')  # (n_bases, n_grid**2)
#         return diffusion
#
#     @partial(jax.jit, static_argnums=(0,))
#     def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
#         return jnp.zeros_like(val)
#
#     @partial(jax.jit, static_argnums=(0,))
#     def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
#         _diffusion_block = self.evaluate_diffusion(val)
#         _zero_block = jnp.zeros_like(_diffusion_block)
#         _diffusion = jnp.block([
#             [_diffusion_block, _zero_block],
#             [_zero_block, _diffusion_block]
#         ])
#         return _diffusion  # (2*n_bases, 2*n_grid**2)
