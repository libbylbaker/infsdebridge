from collections import namedtuple
from .sde import SDE
from .setup import *


def batch_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Batch matrix multiplication"""
    return vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)(A, B)


@partial(jax.jit, static_argnums=(0, ), backend='cpu')
def euler_maruyama(
    sde: SDE,
    initial_vals: jnp.ndarray,
    terminal_vals: jnp.ndarray,
    rng_key: jax.Array = GDRK,
) -> dict:
    """Euler-Maruyama solver for SDEs

    initial_vals: (B, 2*N), complex64
    terminal_vals: (B, 2*N), complex64
    """
    enforce_terminal_constraint = terminal_vals is not None

    SolverState = namedtuple("SolverState", ["vals", "grads", "step_key"])
    init_state = SolverState(
        vals=initial_vals,
        grads=jnp.empty_like(initial_vals),
        step_key=rng_key,
    )

    def euler_maruyama_step(state: SolverState, time: ArrayLike) -> tuple:
        """Euler-Maruyama step, NOTE: all the calculations are over batches"""
        step_key, _ = random.split(state.step_key)
        _drift = vmap(sde.drift, in_axes=(0, None))(state.vals, time)  # (B, 2*N)
        drift_step = _drift * sde.dt
        
        n_batches = state.vals.shape[0]
        _brownian = random.normal(step_key, shape=(n_batches, sde.dim*sde.n_bases**2))  # (B, 2*N^2)
        brownian_step = _brownian * jnp.sqrt(sde.dt)

        _diffusion = vmap(sde.diffusion, in_axes=(0, None))(
            state.vals, time
        )  # (B, 2*N, 2*N^2)
        diffusion_step = batch_matmul(_diffusion, brownian_step)  # (B, 2*N)

        _inv_covariance = vmap(sde.inv_covariance, in_axes=(0, None))(
            state.vals, time
        )  # (B, 2*N, 2*N)

        grads = -batch_matmul(_inv_covariance, diffusion_step) / sde.dt  # (B, 2*N)

        new_vals = state.vals + drift_step + diffusion_step  # (B, 2*N)
        new_state = SolverState(
            vals=new_vals,
            grads=grads,
            step_key=step_key,
        )
        return new_state, (
            state.vals,
            state.grads,
            state.step_key,
        )

    _, (trajectories, gradients, step_keys) = jax.lax.scan(
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
        "last_key": step_keys[-1],
    }
