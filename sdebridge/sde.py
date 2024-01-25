import abc

from jax.numpy import ndarray

from sdebridge.setup import ArrayLike

from .setup import *


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


def matern_kernel(d: jnp.ndarray, sigma: float, rho: float, p: int = 2) -> jnp.ndarray:
    d = jnp.sqrt(jnp.sum(jnp.square(d), axis=-1))
    if p == 0:
        return sigma**2 * jnp.exp(-d / rho)
    elif p == 1:
        return (
            sigma**2 * (1 + jnp.sqrt(3) * d / rho) * jnp.exp(-jnp.sqrt(3) * d / rho)
        )
    elif p == 2:
        return (
            sigma**2
            * (1 + jnp.sqrt(5) * d / rho + 5 * d**2 / (3 * rho**2))
            * jnp.exp(-jnp.sqrt(5) * d / rho)
        )
    else:
        raise NotImplementedError


def matern_Q_kernel(
    landmarks: jnp.ndarray, sigma: float, rho: float, p: int
) -> jnp.ndarray:
    xy_coords = landmarks.reshape(-1, 2)
    num_landmarks = xy_coords.shape[0]
    kernel = matern_kernel(
        d=xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :],
        sigma=sigma,
        rho=rho,
        p=p,
    )
    Q = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
    Q = Q.reshape(2 * num_landmarks, 2 * num_landmarks)
    return Q


class SDE(abc.ABC):
    """Abstract base class for SDEs."""

    def __init__(self, config: ConfigDict):
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
    def dt(self) -> float:
        """time step size"""
        return self.T / self.N

    @property
    def ts(self) -> jnp.ndarray:
        """time steps array (except the starting point)"""
        ts = jnp.linspace(0.0, self.T, self.N + 1)
        return ts

    @abc.abstractmethod
    def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
        """Drift term of the SDE"""
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
        """Diffusion term of the SDE"""
        raise NotImplementedError

    def covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
        """covariance term: \sigma @ \sigma^T"""
        _diffusion = self.diffusion(val, time, **kwargs)
        return jnp.matmul(_diffusion, _diffusion.T)

    def inv_covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
        """inverse covariance term: (\sigma @ \sigma^T)^{-1}"""
        _covariance = self.covariance(val, time, **kwargs)
        return jnp.linalg.pinv(_covariance, hermitian=True, rcond=None)

    def div_covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
        """divergence of covariance term: \nabla_x (\sigma @ \sigma^T)"""
        _jacobian = jax.jacfwd(self.covariance, argnums=0)(val, time, **kwargs)
        return jnp.trace(_jacobian, axis1=-2, axis2=-1)

    def reverse_sde(self, score_func: Callable):
        """Time-invert the SDE using either pre-assigned score p or neural network approximation"""
        config = self.config

        class ReverseSDE(self.__class__):
            def __init__(self):
                super().__init__(config)

            def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
                inverted_time = self.T - time
                rev_drift_term = super().drift(val=val, time=inverted_time, **kwargs)

                _score = score_func(val=val, time=inverted_time, **kwargs)
                _covariance = super().covariance(val=val, time=inverted_time, **kwargs)
                score_term = jnp.dot(_covariance, _score)

                div_term = super().div_covariance(val=val, time=inverted_time, **kwargs)
                return -rev_drift_term + score_term + div_term

            def diffusion(self, val: ArrayLike, time: ArrayLike) -> jnp.ndarray:
                inverted_time = self.T - time
                return super().diffusion(val=val, time=inverted_time)

        return ReverseSDE()

    def bridge_sde(self, score_func: Callable):
        """Bridge the SDE using either pre-assigned score h or neural network approximation"""
        config = self.config

        class BridgeSDE(self.__class__):
            def __init__(self):
                super().__init__(config)

            def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jnp.ndarray:
                orig_drift_term = super().drift(val=val, time=time, **kwargs)

                _score = score_func(val=val, time=time, **kwargs)
                _covariance = super().covariance(val=val, time=time, **kwargs)
                score_term = jnp.dot(_covariance, _score)
                return orig_drift_term + score_term

            def diffusion(
                self, val: ArrayLike, time: ArrayLike, **kwargs
            ) -> jnp.ndarray:
                return super().diffusion(val=val, time=time, **kwargs)

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

    def drift(self, val: ArrayLike, time: ArrayLike) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: ArrayLike, time: ArrayLike) -> jnp.ndarray:
        return self.alpha * jnp.eye(self.dim)

    def covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> ndarray:
        return jnp.dot(self.diffusion(val, time), self.diffusion(val, time).T)


class DampedBrownianSDE(SDE):
    def __init__(self, config):
        super().__init__(config)

    @property
    def alpha(self) -> float:
        return self.config.alpha

    @alpha.setter
    def alpha(self, value: float):
        self.config.alpha = value

    def drift(self, val: ArrayLike, time: ArrayLike) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: ArrayLike, time: ArrayLike) -> jnp.ndarray:
        return self.alpha * jnp.diag(
            1.0
            / jnp.concatenate(
                [jnp.arange(1, self.dim // 2 + 1), jnp.arange(1, self.dim // 2 + 1)]
            )
        )

    def covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> ndarray:
        return jnp.dot(self.diffusion(val, time), self.diffusion(val, time).T)


class GaussianKernelSDE(SDE):
    def __init__(self, config: ConfigDict):
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

    def drift(self, val: jnp.ndarray, time: ArrayLike) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: jnp.ndarray, time: ArrayLike) -> jnp.ndarray:
        return gaussian_Q_half_kernel(landmarks=val, alpha=self.alpha, sigma=self.sigma)

    def covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> ndarray:
        return gaussian_Q_kernel(landmarks=val, alpha=self.alpha, sigma=self.sigma)


class MaternKernelSDE(SDE):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

    @property
    def sigma(self) -> float:
        return self.config.sigma

    @sigma.setter
    def sigma(self, value: float):
        self.config.sigma = value

    @property
    def rho(self) -> float:
        return self.config.rho

    @rho.setter
    def rho(self, value: float):
        self.config.rho = value

    @property
    def p(self) -> int:
        return self.config.p

    @p.setter
    def p(self, value: int):
        self.config.p = value

    def drift(self, val: jnp.ndarray, time: ArrayLike) -> jnp.ndarray:
        return jnp.zeros_like(val)

    def diffusion(self, val: jnp.ndarray, time: ArrayLike) -> jnp.ndarray:
        return matern_Q_kernel(landmarks=val, sigma=self.sigma, rho=self.rho, p=self.p)


class CustomizedSDE(SDE):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

    def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> ndarray:
        return self.config.drift(val=val, time=time, **kwargs)

    def diffusion(self, val: ArrayLike, time: ArrayLike, **kwargs) -> ndarray:
        return self.config.diffusion(val=val, time=time, **kwargs)

    def covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> ndarray:
        return self.config.covariance(val=val, time=time, **kwargs)
