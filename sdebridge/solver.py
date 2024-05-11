import jax
import jax.numpy as jnp


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
