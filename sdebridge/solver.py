from collections import namedtuple

from jax.tree_util import Partial

from .sde import SDE
from .setup import *


def batch_multi(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Batch matrix multiplication"""
    return vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(A, B)


@Partial(jax.jit, static_argnames=("sde"))
def euler_maruyama(
    sde: SDE,
    initial_vals: jnp.ndarray,
    terminal_vals: jnp.ndarray,
    rng_key: jax.Array = random.PRNGKey(0),
) -> dict:
    """Euler-Maruyama solver for SDEs"""
    enforce_terminal_constraint = terminal_vals is not None
    assert initial_vals.shape[-1] == sde.dim
    assert terminal_vals.shape[-1] == sde.dim if enforce_terminal_constraint else True

    SolverState = namedtuple(
        "SolverState", ["vals", "scaled_stochastic_increment", "step_key"]
    )
    init_state = SolverState(
        vals=initial_vals,
        scaled_stochastic_increment=jnp.empty_like(initial_vals),
        step_key=rng_key,
    )

    def euler_maruyama_step(state: SolverState, time: ArrayLike) -> tuple:
        """Euler-Maruyama step, NOTE: all the calculations are over batches"""
        step_key, _ = random.split(state.step_key)
        _drift = vmap(sde.drift, in_axes=(0, None))(state.vals, time)  # (B, d)
        drift_step = _drift * sde.dt

        _brownian = random.normal(step_key, shape=state.vals.shape)  # (B, d)
        brownian_step = _brownian * jnp.sqrt(sde.dt)

        _diffusion = vmap(sde.diffusion, in_axes=(0, None))(
            state.vals, time
        )  # (B, d, d)
        diffusion_step = batch_multi(_diffusion, brownian_step)  # (B, d)

        _inv_covariance = vmap(sde.inv_covariance, in_axes=(0, None))(
            state.vals, time
        )  # (B, d, d)

        scaled_stochastic_increment = (
            -batch_multi(_inv_covariance, diffusion_step) / sde.dt
        )  # (B, d)

        new_vals = state.vals + drift_step + diffusion_step  # (B, d)
        new_state = SolverState(
            vals=new_vals,
            scaled_stochastic_increment=scaled_stochastic_increment,
            step_key=step_key,
        )
        return new_state, (
            state.vals,
            state.scaled_stochastic_increment,
            state.step_key,
        )

    _, (trajectories, scaled_stochastic_increments, step_keys) = jax.lax.scan(
        euler_maruyama_step,
        init=init_state,
        xs=(sde.ts[:-1]),
        length=sde.N,
    )

    if enforce_terminal_constraint:
        trajectories = trajectories.at[-1].set(terminal_vals)
    return {
        "trajectories": jnp.swapaxes(trajectories, 0, 1),
        "scaled_stochastic_increments": jnp.swapaxes(
            scaled_stochastic_increments, 0, 1
        ),
        "step_keys": step_keys,
    }
