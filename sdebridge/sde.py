import abc
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange


### Kernels ###
def gaussian_Q_half_kernel(
    landmarks: jnp.ndarray, alpha: float, sigma: float
) -> jnp.ndarray:
    xy_coords = landmarks.reshape(-1, 2)
    num_landmarks = xy_coords.shape[0]
    diff = xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :]
    dis = jnp.sum(jnp.square(diff), axis=-1)
    kernel = alpha * jnp.exp(-dis / sigma**2)
    Q_half = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
    Q_half = Q_half.reshape(2 * num_landmarks, 2 * num_landmarks)
    return Q_half


def gaussian_Q_kernel(
    landmarks: jnp.ndarray, alpha: float, sigma: float
) -> jnp.ndarray:
    xy_coords = landmarks.reshape(-1, 2)
    num_landmarks = xy_coords.shape[0]
    diff = xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :]
    dis = jnp.sum(jnp.square(diff), axis=-1)
    kernel = (
        0.5 * (alpha**4) * (sigma**2) * jnp.pi * jnp.exp(-0.5 * dis / (sigma**2))
    )
    Q = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
    Q = Q.reshape(2 * num_landmarks, 2 * num_landmarks)
    return Q

def gaussian_kernel_2d(x: jnp.ndarray, y: jnp.ndarray, alpha: float, sigma: float) -> jnp.ndarray:
    return alpha * jnp.exp(-jnp.linalg.norm(x - y, axis=-1) ** 2 / (2 * sigma ** 2))

class SDE(abc.ABC):
    """Abstract base class for SDEs."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def T(self) -> float:
        """total time"""
        return self.config.T

    @T.setter
    def T(self, value: float):
        self.config.T = value

    @property
    def N(self) -> int:
        """number of time steps"""
        return self.config.N

    @N.setter
    def N(self, value: int):
        self.config.N = value

    @property
    def dim(self) -> int:
        """dimension"""
        return self.config.dim

    @dim.setter
    def dim(self, value: int):
        self.config.dim = value

    @property
    def n_bases(self) -> int:
        """number of bases"""
        return self.config.n_bases
    
    @n_bases.setter
    def n_bases(self, value: int):
        self.config.n_bases = value

    @property
    def dt(self) -> float:
        """time step size"""
        return self.T / self.N

    @property
    def ts(self) -> jnp.ndarray:
        """time steps array (except the starting point)"""
        ts = jnp.linspace(0.0, self.T, self.N + 1)
        return ts

    @abc.abstractmethod
    def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """Drift term of the SDE"""
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """Diffusion term of the SDE"""
        raise NotImplementedError

    def covariance(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """covariance term: \sigma @ \sigma^T"""
        _diffusion = self.diffusion(val, time)
        return jnp.matmul(_diffusion, jnp.conj(_diffusion).T)

    def div_covariance(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """divergence of covariance term: \nabla_x (\sigma @ \sigma^T)"""
        _jacobian = jax.jacfwd(self.covariance, argnums=0, holomorphic=True)(val, time)
        return jnp.trace(_jacobian, axis1=-2, axis2=-1)

    def reverse_sde(self, score_func):
        """Time-invert the SDE using either pre-assigned score p or neural network approximation"""
        config = self.config

        class ReverseSDE(self.__class__):
            def __init__(self):
                super().__init__(config)

            def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
                inverted_time = self.T - time
                rev_drift_term = super().drift(val=val, time=inverted_time)

                _score = jnp.squeeze(score_func(val=val, time=inverted_time))
                _covariance = super().covariance(val=val, time=inverted_time)
                score_term = jnp.dot(_covariance, _score)

                div_term = super().div_covariance(val=val, time=inverted_time)
                return -rev_drift_term + score_term + div_term 
            
            def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
                inverted_time = self.T - time
                return super().diffusion(val=val, time=inverted_time)

        return ReverseSDE()

    def bridge_sde(self, score_func):
        """Bridge the SDE using either pre-assigned score h or neural network approximation"""
        config = self.config

        class BridgeSDE(self.__class__):
            def __init__(self):
                super().__init__(config)

            def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
                orig_drift_term = super().drift(val=val, time=time)

                _score = jnp.squeeze(score_func(val=val, time=time))
                _covariance = super().covariance(val=val, time=time)
                score_term = jnp.dot(_covariance, _score)
                return orig_drift_term + score_term

            def diffusion(
                self, val: jnp.ndarray, time: jnp.ndarray
            ) -> jnp.ndarray:
                return super().diffusion(val=val, time=time)

        return BridgeSDE()


class BrownianSDE(SDE):
    def __init__(self, config):
        super().__init__(config)

    @property
    def alpha(self) -> float:
        return self.config.alpha

    @alpha.setter
    def alpha(self, value: float):
        self.config.alpha = value

    def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return self.alpha * jnp.eye(self.dim)


class DampedBrownianSDE(SDE):
    def __init__(self, config):
        super().__init__(config)

    @property
    def alpha(self) -> float:
        return self.config.alpha

    @alpha.setter
    def alpha(self, value: float):
        self.config.alpha = value

    def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return self.alpha * jnp.diag(
            1.0
            / jnp.concatenate(
                [jnp.arange(1, self.dim // 2 + 1), jnp.arange(1, self.dim // 2 + 1)]
            )
        )


class GaussianKernelSDE(SDE):
    def __init__(self, config):
        super().__init__(config)

    @property
    def alpha(self) -> float:
        return self.config.alpha

    @alpha.setter
    def alpha(self, value: float):
        self.config.alpha = value

    @property
    def sigma(self) -> float:
        return self.config.sigma

    @sigma.setter
    def sigma(self, value: float):
        self.config.sigma = value

    def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return gaussian_Q_half_kernel(landmarks=val, alpha=self.alpha, sigma=self.sigma)



class FourierGaussianKernelSDE(SDE):
    def __init__(self, config):
        super().__init__(config)
        self._fourier_basis = self.fourier_basis()
        self._grid = self.grid()
        self._init_S_eval = self.evaluate_S(self.init_S)


    @property
    def alpha(self) -> float:
        return self.config.alpha

    @property
    def sigma(self) -> float:
        return self.config.sigma

    @property
    def init_S(self) -> jnp.ndarray:
        return self.config.init_S
    
    @property
    def n_samples(self) -> int:
        return self.config.init_S.shape[0]
    
    @property
    def n_bases(self) -> int:
        return self.config.n_bases
    
    @property
    def n_padding(self) -> int:
        return (self.n_samples - self.n_bases) // 2
    
    @property
    def n_grid(self) -> int:
        return self.config.n_grid

    # @property
    def fourier_basis(self) -> jnp.ndarray:
        base_fn = lambda freq: jnp.exp(1j * jnp.arange(-self.n_samples//2, self.n_samples//2) * freq)
        freqs = jnp.fft.fftshift(
            jnp.fft.fftfreq(self.n_bases, d=1/(2.0*jnp.pi))
        )
        return jax.vmap(base_fn)(freqs) # (n_bases, n_samples)
    
    # @property
    def grid(self) -> jnp.ndarray:
        grid_range = self.config.grid_range
        grid = jnp.linspace(grid_range[0], grid_range[1], self.n_grid)
        grid = jnp.stack(jnp.meshgrid(grid, grid, indexing='xy'), axis=-1)
        return grid     # (n_grid, n_grid, 2)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_S(self, S: jnp.ndarray) -> jnp.ndarray:
        S_coeffs = jnp.fft.fft(S, n=self.n_samples, axis=0) / jnp.sqrt(self.n_samples)
        S_coeffs = jnp.fft.fftshift(S_coeffs, axes=0)   # (n_samples, 2)
        S_eval = jnp.matmul(self._fourier_basis, S_coeffs) / jnp.sqrt(self.n_samples) # (n_bases, 2)
        return S_eval

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_X(self, val: jnp.ndarray) -> jnp.ndarray:
        X_coeffs = jnp.stack(jnp.split(val, 2, axis=-1), axis=-1)   # (n_bases, 2)
        X_coeffs = X_coeffs / jnp.sqrt(self.n_samples)
        X_coeffs = jnp.pad(X_coeffs, ((self.n_padding, self.n_padding), (0, 0)), mode='constant', constant_values=0)    # (n_samples, 2)
        X_eval = jnp.matmul(self._fourier_basis, X_coeffs) / jnp.sqrt(self.n_samples) # (n_bases, 2)
        return X_eval
    
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_Q(self, val: jnp.ndarray) -> jnp.ndarray:
        X_eval = self.evaluate_X(val)
        S_eval = X_eval + self._init_S_eval  # (n_bases, 2)
        Q_eval = jax.vmap(
            jax.vmap(
                jax.vmap(
                    partial(gaussian_kernel_2d, alpha=self.alpha, sigma=self.sigma),
                    in_axes=(None, 0),
                    out_axes=0
                ),
                in_axes=(None, 1),
                out_axes=1
            ),
            in_axes=(0, None),
            out_axes=0
        )(S_eval, self._grid)    # (n_bases, n_grid, n_grid)
        return Q_eval
    
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_diffusion(self, val: jnp.ndarray) -> jnp.ndarray:
        Q_eval = self.evaluate_Q(val)
        diffusion = jnp.fft.fft2(Q_eval, axes=(1, 2), norm='ortho')  # (n_bases, n_grid, n_grid)
        diffusion = jnp.fft.ifft(diffusion, axis=0, norm='ortho')  # (n_bases, n_grid, n_grid
        diffusion = rearrange(diffusion, 'b g1 g2 -> b (g1 g2)')   # (n_bases, n_grid**2)
        return diffusion
    
    @partial(jax.jit, static_argnums=(0,))
    def drift(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)
    
    @partial(jax.jit, static_argnums=(0,))
    def diffusion(self, val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        _diffusion_block = self.evaluate_diffusion(val)
        _zero_block = jnp.zeros_like(_diffusion_block)
        _diffusion = jnp.block([
            [_diffusion_block, _zero_block], 
            [_zero_block, _diffusion_block]
        ])
        return _diffusion       # (2*n_bases, 2*n_grid**2)