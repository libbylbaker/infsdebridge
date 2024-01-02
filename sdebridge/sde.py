import abc

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .utils import Partial, batch_multi, eval_Q


class SDE(abc.ABC):
    """Abstract base class for SDEs."""

    def __init__(self, sde_params: dict):
        super().__init__()
        assert "dimension" in sde_params.keys() and "num_steps" in sde_params.keys()
        self.params = sde_params

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """total time"""
        raise NotImplementedError

    @property
    def N(self) -> int:
        """number of time steps"""
        return self.params["num_steps"]

    @property
    def dim(self) -> int:
        """dimension"""
        return self.params["dimension"]

    @property
    def dt(self) -> float:
        """time step size"""
        return self.T / self.N

    @property
    def ts(self) -> jax.Array:
        """time steps array (except the starting point)"""
        ts = jnp.linspace(0.0, self.T, self.N + 1)
        return ts

    @abc.abstractmethod
    def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """Drift term of the SDE"""
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """Diffusion term of the SDE"""
        raise NotImplementedError

    def inv_diffusion(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """inverse of diffusion term"""
        return jnp.linalg.inv(self.diffusion(val, time, **kwargs))

    def covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """covariance term: \sigma @ \sigma^T"""
        _diffusion = self.diffusion(val, time, **kwargs)
        return jnp.dot(_diffusion, _diffusion.T)

    def inv_covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """inverse covariance term: (\sigma @ \sigma^T)^{-1}"""
        return jnp.linalg.inv(self.covariance(val, time, **kwargs))

    def div_covariance(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """divergence of covariance term: \nabla_x (\sigma @ \sigma^T)"""
        _jacobian_covariance = jax.jacfwd(self.covariance, argnums=0)(
            val, time, **kwargs
        )
        return jnp.trace(_jacobian_covariance)

    @abc.abstractmethod
    def score_p_density(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """score of the transition density"""
        raise NotImplementedError

    @abc.abstractmethod
    def score_h_density(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
        """score of Doob's h-transform density, used for simulating the bridge directly"""
        raise NotImplementedError

    def reverse_sde(self, score_p_density: callable):
        """Time-invert the SDE using either pre-assigned score p or neural network approximation"""
        if score_p_density is None:
            score_p_density = self.score_p_density

        sde_params = self.params

        class ReverseSDE(self.__class__):
            def __init__(self):
                super().__init__(sde_params=sde_params)

            def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
                inverted_time = self.T - time
                rev_drift = super().drift(val=val, time=inverted_time, **kwargs)
                score = score_p_density(val=val, time=inverted_time, **kwargs)
                score_term = batch_multi(
                    super().covariance(val=val, time=inverted_time), score
                )
                div_term = super().div_covariance(val=val, time=inverted_time)
                return rev_drift + score_term + div_term

            def diffusion(self, val: ArrayLike, time: ArrayLike) -> jax.Array:
                inverted_time = self.T - time
                return super().diffusion(val=val, time=inverted_time)

        return ReverseSDE()

    def bridge_sde(self, score_h_density: callable):
        """Bridge the SDE using either pre-assigned score h or neural network approximation"""
        if score_h_density is None:
            score_h_density = self.score_h_density

        sde_params = self.params

        class BridgeSDE(self.__class__):
            def __init__(self):
                super().__init__(sde_params=sde_params)

            def drift(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
                orig_drift = super().drift(val=val, time=time, **kwargs)
                score = score_h_density(val=val, time=time, **kwargs)
                score_term = batch_multi(super().covariance(val=val, time=time), score)
                return orig_drift + score_term

            def diffusion(self, val: ArrayLike, time: ArrayLike, **kwargs) -> jax.Array:
                return super().diffusion(val=val, time=time, **kwargs)

        return BridgeSDE()


class BrownianSDE(SDE):
    def __init__(self, sde_params: dict):
        super().__init__(sde_params=sde_params)

    @property
    def T(self) -> float:
        return 1.0

    def drift(self, val: ArrayLike, time: ArrayLike) -> jax.Array:
        return jnp.zeros_like(val)

    def diffusion(self, val: ArrayLike, time: ArrayLike) -> jax.Array:
        return jnp.eye(self.dim)

    def score_p_density(
        self, val: ArrayLike, time: ArrayLike, init_val: ArrayLike, term_val: ArrayLike
    ) -> jax.Array:
        assert val.shape == init_val.shape
        return -(val - init_val) / (time + 1e-4)

    def score_h_density(
        self, val: ArrayLike, time: ArrayLike, init_val: ArrayLike, term_val: ArrayLike
    ) -> jax.Array:
        assert val.shape == term_val.shape
        return -(val - term_val) / (time - self.T + 1e-4)


class Fixed_QSDE(SDE):
    def __init__(self, sde_params: dict):
        super().__init__(sde_params=sde_params)
        assert "init_Q" in sde_params.keys()
        assert self.params["init_Q"].shape[0] == self.dim

    @property
    def Q(self) -> jax.Array:
        return self.params["init_Q"]

    @property
    def T(self) -> float:
        return 1.0

    def drift(self, val: jax.Array, time: float) -> jax.Array:
        return jnp.zeros_like(val)

    def diffusion(self, val: jax.Array, time: float) -> jax.Array:
        return self.Q

    def score_p_density(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        return super().score_p_density(val, time, **kwargs)

    def score_h_density(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        return super().score_h_density(val, time, **kwargs)


class QSDE(SDE):
    def __init__(self, sde_params: dict):
        super().__init__(sde_params=sde_params)
        assert "alpha" in sde_params.keys() and "sigma" in sde_params.keys()

    @property
    def alpha(self) -> float:
        return self.params["alpha"]

    @property
    def sigma(self) -> float:
        return self.params["sigma"]

    @property
    def T(self) -> float:
        return 1.0

    def drift(self, val: jax.Array, time: float) -> jax.Array:
        return jnp.zeros_like(val)

    def diffusion(self, val: jax.Array, time: float) -> jax.Array:
        return jax.vmap(eval_Q, in_axes=(0, None, None), out_axes=0)(
            val, self.alpha, self.sigma
        )

    def score_p_density(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        return super().score_p_density(val, time, **kwargs)

    def score_h_density(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        return super().score_h_density(val, time, **kwargs)
