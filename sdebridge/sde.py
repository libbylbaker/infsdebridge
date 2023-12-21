import abc

import jax
import jax.numpy as jnp

from .utils import batch_multi


class SDE(abc.ABC):
    """Abstract base class for SDEs."""

    def __init__(self, dimension: int, num_steps: int):
        super().__init__()
        self.d = dimension
        self.N = num_steps

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """total time"""
        raise NotImplementedError

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
    def drift(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        """drift term"""
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion(self, val: jax.Array, time: float) -> jax.Array:
        """diffusion term"""
        raise NotImplementedError

    def inv_diffusion(self, val: jax.Array, time: float) -> jax.Array:
        """inverse of diffusion term"""
        return jnp.linalg.inv(self.diffusion(val, time))

    def covariance(self, val: jax.Array, time: float) -> jax.Array:
        """covariance term: \sigma @ \sigma^T"""
        return jnp.dot(self.diffusion(val, time), self.diffusion(val, time).T)

    def inv_covariance(self, val: jax.Array, time: float) -> jax.Array:
        """inverse covariance term: (\sigma @ \sigma^T)^{-1}"""
        return jnp.linalg.inv(self.covariance(val, time))

    def div_covariance(self, val: jax.Array, time: float) -> jax.Array:
        """divergence of covariance term: \nabla_x (\sigma @ \sigma^T)"""
        jacobian_covariance = jax.jacfwd(self.covariance, argnums=0)(val, time)
        return jnp.trace(jacobian_covariance)

    @abc.abstractmethod
    def score_p_density(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        """score of the transition density"""
        raise NotImplementedError

    @abc.abstractmethod
    def score_h_density(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        """score of Doob's h-transform density, used for simulating the bridge directly"""
        raise NotImplementedError

    def reverse_sde(self, score_p_density: callable):
        """Time-invert the SDE using either pre-assigned score p or neural network approximation"""
        if score_p_density is None:
            score_p_density = self.score_p_density

        d = self.d
        N = self.N

        class ReverseSDE(self.__class__):
            def __init__(self):
                super().__init__(dimension=d, num_steps=N)

            def drift(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
                inverted_time = self.T - time
                rev_drift = super().drift(val=val, time=inverted_time, **kwargs)
                score = score_p_density(val=val, time=inverted_time, **kwargs)
                score_term = batch_multi(
                    super().covariance(val=val, time=inverted_time), score
                )
                div_term = super().div_covariance(val=val, time=inverted_time)
                return rev_drift + score_term + div_term

            def diffusion(self, val: jax.Array, time: float) -> jax.Array:
                inverted_time = self.T - time
                return super().diffusion(val=val, time=inverted_time)

        return ReverseSDE()

    def bridge_sde(self, score_h_density: callable):
        """Bridge the SDE using either pre-assigned score h or neural network approximation"""
        if score_h_density is None:
            score_h_density = self.score_h_density

        d = self.d
        N = self.N

        class BridgeSDE(self.__class__):
            def __init__(self):
                super().__init__(dimension=d, num_steps=N)

            def drift(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
                orig_drift = super().drift(val=val, time=time, **kwargs)
                score = score_h_density(val=val, time=time, **kwargs)
                score_term = batch_multi(super().covariance(val=val, time=time), score)
                return orig_drift + score_term

            def diffusion(self, val: jax.Array, time: float) -> jax.Array:
                return super().diffusion(val=val, time=time)

        return BridgeSDE()


class BrownianSDE(SDE):
    def __init__(self, dimension: int, num_steps: int):
        super().__init__(dimension, num_steps)

    @property
    def T(self) -> float:
        return 1.0

    def drift(self, val: jax.Array, time: float) -> jax.Array:
        return jnp.zeros_like(val)

    def diffusion(self, val: jax.Array, time: float) -> jax.Array:
        return jnp.eye(self.d)

    def score_p_density(
        self, val: jax.Array, time: float, init_val: jax.Array, term_val: jax.Array
    ) -> jax.Array:
        return -(val - init_val) / (time + 1e-4)

    def score_h_density(
        self, val: jax.Array, time: float, init_val: jax.Array, term_val: jax.Array
    ) -> jax.Array:
        return -(val - term_val) / (time - self.T + 1e-4)


class QSDE(SDE):
    def __init__(self, dimension: int, num_steps: int):
        super().__init__(dimension, num_steps)

    @property
    def T(self) -> float:
        return 1.0

    def drift(self, val: jax.Array, time: float) -> jax.Array:
        return jnp.zeros_like(val)

    # todo: implement Q matrix here.
    def diffusion(self, val: jax.Array, time: float, **kwargs) -> jax.Array:
        return super().diffusion(val, time, **kwargs)
