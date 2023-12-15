from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state


### Dimension helpers
def flatten_batch(x: jax.Array):
    assert len(x.shape) >= 2
    return x.reshape(-1, x.shape[-1])


def unsqueeze(x: jax.Array, axis: int):
    return jnp.expand_dims(x, axis=axis)


def squeeze(x: jax.Array, axis: int):
    return jnp.squeeze(x, axis=axis)


def batch_multi(a: jax.Array, b: jax.Array) -> jax.Array:
    """Perform matrix multiplication for a single matrix over each matrix in a batch."""
    return jax.vmap(lambda x, y: jnp.dot(x, y), in_axes=(None, 0), out_axes=0)(a, b)


### Network helpers
@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    batch_stats: any
    metrics: Metrics


def create_train_state(
    module: nn.Module, rng: jax.Array, learning_rate: float, input_shapes: list
) -> TrainState:
    init_inputs = [jnp.zeros(shape=shape) for shape in input_shapes]
    variables = module.init(rng, *init_inputs, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}

    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        metrics=Metrics.empty(),
    )


@partial(jax.jit, static_argnums=(0,))
def eval_score(state: TrainState, val: jax.Array, time: float) -> jax.Array:
    assert len(val.shape) == 2  # (B, d)
    time = jnp.tile(time, reps=(val.shape[0], 1))  # (B, 1)
    return state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        val,
        time,
        train=False,
    )


def get_iterable_dataset(generator: callable, dtype: any, shape: any):
    if type(dtype) == tf.DType and type(shape) == list:
        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=(tf.TensorSpec(shape=shape, dtype=dtype))
        )
    elif type(dtype) == tuple and type(shape) == list:
        assert len(dtype) == len(shape)
        signatures = tuple(
            [tf.TensorSpec(shape=shape[i], dtype=dtype[i]) for i in range(len(dtype))]
        )
        dataset = tf.data.Dataset.from_generator(generator, output_signature=signatures)
    else:
        raise ValueError("Invalid dtype or shape")
    iterable_dataset = iter(tfds.as_numpy(dataset))
    return iterable_dataset


### SDE helpers
@partial(jax.jit, static_argnums=(0,))
def euler_maruyama(
    sde, initial_val: jax.Array, rng: jax.Array, terminal_val: jax.Array = None
) -> dict:
    """Euler-Maruyama solver for SDEs"""
    enforce_terminal_constraint = terminal_val is not None
    assert initial_val.shape[-1] == sde.d
    assert terminal_val.shape[-1] == sde.d if enforce_terminal_constraint else True
    SolverState = namedtuple("SolverState", ["val", "scaled_stochastic", "rng"])
    init_state = SolverState(
        val=initial_val, scaled_stochastic=jnp.empty_like(initial_val), rng=rng
    )

    def euler_maruyama_step(state: SolverState, time: float) -> tuple:
        """Euler-Maruyama step"""
        new_rng, _ = jax.random.split(state.rng)
        drift_step = sde.drift(state.val, time) * sde.dt
        brownian_step = jnp.sqrt(sde.dt) * jax.random.normal(
            new_rng, shape=state.val.shape
        )
        diffusion_step = batch_multi(sde.diffusion(state.val, time), brownian_step)
        inv_covariance = sde.inv_covariance(state.val, time)
        scaled_stochastic = -batch_multi(inv_covariance / sde.dt, diffusion_step)
        new_val = state.val + drift_step + diffusion_step
        new_state = SolverState(
            val=new_val, scaled_stochastic=scaled_stochastic, rng=new_rng
        )
        return new_state, (state.val, state.scaled_stochastic, state.rng)

    _, (trajectories, scaled_stochastics, rng) = jax.lax.scan(
        euler_maruyama_step, init=init_state, xs=(sde.ts)
    )

    if enforce_terminal_constraint:
        trajectories = trajectories.at[-1].set(terminal_val)
    return {
        "trajectories": jnp.swapaxes(trajectories, 0, 1),
        "scaled_stochastics": jnp.swapaxes(scaled_stochastics, 0, 1),
        "rng": rng,
    }


### Plotting helpers
def plot_2d_vector_field(
    X: callable,
    X_ref: callable,
    xs: jax.Array,
    ts: jax.Array,
    suptitle: str,
    scale: float = None,
    **kwargs,
):
    xm, ym = jnp.meshgrid(xs, xs)
    x = jnp.stack([xm.flatten(), ym.flatten()], axis=-1)
    fig, ax = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4))
    X_ref = partial(X_ref, **kwargs) if X_ref is not None else None
    for i, t in enumerate(ts):
        vector_field = X(x, t) if X is not None else None
        vector_field_ref = X_ref(x, t) if X_ref is not None else None
        if vector_field is not None:
            u = vector_field[:, 0].reshape(xm.shape)
            v = vector_field[:, 1].reshape(xm.shape)
            ax[i].quiver(xm, ym, u, v, color="b", scale=scale)

        if vector_field_ref is not None:
            u_ref = vector_field_ref[:, 0].reshape(xm.shape)
            v_ref = vector_field_ref[:, 1].reshape(xm.shape)
            ax[i].quiver(xm, ym, u_ref, v_ref, color="r", scale=scale)

        ax[i].set_title(f"t = {t:.1f}")
    fig.suptitle(suptitle)
    plt.show()


def plot_2d_trajectories(trajectories: jax.Array, title: str, **kwargs):
    colormap = plt.cm.get_cmap("spring")
    num_trajectories = trajectories.shape[0]
    colors = [colormap(i) for i in jnp.linspace(0, 1, num_trajectories)]
    for i in range(num_trajectories):
        plt.plot(
            trajectories[i, :, 0],
            trajectories[i, :, 1],
            color=colors[i],
            zorder=1,
            alpha=0.5,
            **kwargs,
        )
        plt.scatter(
            trajectories[i, 1, 0],
            trajectories[i, 1, 1],
            color="b",
            marker="o",
            edgecolors="k",
            zorder=2,
        )
        plt.scatter(
            trajectories[i, -2, 0],
            trajectories[i, -2, 1],
            color="c",
            marker="D",
            edgecolors="k",
            zorder=2,
        )
    plt.title(title)


def plot_trajectories(trajectories: jax.Array, title: str, **kwargs):
    colormap = plt.cm.get_cmap("spring")
    assert len(trajectories.shape) == 3
    num_trajectories = trajectories.shape[0]
    dim = trajectories.shape[2]
    colors = [colormap(i) for i in jnp.linspace(0, 1, num_trajectories)]
    plt.figure(figsize=(8, 8))
    for i in range(num_trajectories):
        for j in range(dim // 2):
            plt.plot(
                trajectories[i, :, 2 * j],
                trajectories[i, :, 2 * j + 1],
                color=colors[i],
                zorder=1,
                alpha=0.2,
                **kwargs,
            )
            plt.scatter(
                trajectories[i, 0, 2 * j],
                trajectories[i, 0, 2 * j + 1],
                color="b",
                marker="o",
                edgecolors="k",
                zorder=2,
            )
            plt.scatter(
                trajectories[i, -1, 2 * j],
                trajectories[i, -1, 2 * j + 1],
                color=colors[i],
                marker="D",
                edgecolors="k",
                zorder=2,
            )
    plt.axis("equal")
    plt.title(title)


def plot_single_trajectory(trajectory: jax.Array, title: str, **kwargs):
    colormap = plt.cm.get_cmap("jet")
    assert len(trajectory.shape) == 2
    dim = trajectory.shape[-1]
    colors = [colormap(i) for i in jnp.linspace(0, 1, dim // 2)]
    plt.figure(figsize=(8, 8))
    for i in range(dim // 2):
        plt.plot(
            trajectory[:, 2 * i],
            trajectory[:, 2 * i + 1],
            color=colors[i],
            zorder=1,
            alpha=0.5,
            **kwargs,
        )
        plt.scatter(
            trajectory[0, 2 * i],
            trajectory[0, 2 * i + 1],
            color=colors[i],
            marker="o",
            edgecolors="k",
            zorder=2,
        )
        plt.scatter(
            trajectory[-1, 2 * i],
            trajectory[-1, 2 * i + 1],
            color=colors[i],
            marker="D",
            edgecolors="k",
            zorder=2,
        )
    plt.axis("equal")
    plt.title(title)


### Data helpers
def sample_circle(num_points: int, scale: float, shifts: jax.Array) -> jax.Array:
    theta = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False)
    x = jnp.cos(theta)
    y = jnp.sin(theta)
    return (scale * jnp.stack([x, y], axis=1) + shifts).flatten()


# todo: The order of sampled points needs to be checked.
def sample_square(num_points: int, scale: float, shifts: jax.Array) -> jax.Array:
    num_points_per_side = num_points // 4
    x1 = jnp.linspace(-1, 1, num_points_per_side, endpoint=False)
    x2 = jnp.linspace(1, -1, num_points_per_side, endpoint=False)
    y1 = jnp.linspace(-1, 1, num_points_per_side, endpoint=False)
    y2 = jnp.linspace(1, -1, num_points_per_side, endpoint=False)
    xy1 = jnp.stack([x1, jnp.ones_like(x1)], axis=1)
    xy2 = jnp.stack([jnp.ones_like(y1), y2], axis=1)
    xy3 = jnp.stack([x2, -jnp.ones_like(x2)], axis=1)
    xy4 = jnp.stack([-jnp.ones_like(y1), y1], axis=1)
    return (scale * jnp.concatenate([xy1, xy2, xy3, xy4], axis=0) + shifts).flatten()
