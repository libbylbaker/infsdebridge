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
from jax.tree_util import Partial
from jax.typing import ArrayLike


### Dimension helpers
def flatten_batch(x: jax.Array):
    assert len(x.shape) >= 2
    return x.reshape(-1, x.shape[-1])


def unsqueeze(x: jax.Array, axis: int):
    return jnp.expand_dims(x, axis=axis)


def squeeze(x: jax.Array, axis: int):
    return jnp.squeeze(x, axis=axis)


def batch_batch_multi(a: jax.Array, b: jax.Array) -> jax.Array:
    """Perform matrix multiplication for batched matrices."""
    return jax.vmap(lambda x, y: jnp.dot(x, y), in_axes=(0, 0), out_axes=0)(a, b)


def single_batch_multi(a: jax.Array, b: jax.Array) -> jax.Array:
    """Perform matrix multiplication for batched matrix and single matrix."""
    return jax.vmap(lambda x, y: jnp.dot(x, y), in_axes=(0, 0), out_axes=0)(a, b)


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


@jax.jit
def eval_score(state: TrainState, val: jax.Array, time: ArrayLike) -> jax.Array:
    assert len(val.shape) == 2, f"Invalid shape: {val.shape}"
    time = jnp.tile(time, (val.shape[0], 1))
    score = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        val,
        time,
        train=False,
    )
    return score


def get_iterable_dataset(generator: callable, dtype: any, shape: any):
    if type(dtype) == tf.DType and type(shape) == list:
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(tf.TensorSpec(shape=shape, dtype=dtype)),
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


def get_next_batch(generator: callable, rng: jax.Array):
    new_rng, _ = jax.random.split(rng)
    data = next(generator(new_rng))
    return data, new_rng


### SDE helpers
@Partial(jax.jit, static_argnames=("sde"))
def euler_maruyama(
    sde, initial_vals: jax.Array, rng: jax.Array, terminal_vals: jax.Array = None
) -> dict:
    """Euler-Maruyama solver for SDEs"""
    enforce_terminal_constraint = terminal_vals is not None
    assert initial_vals.shape[-1] == sde.dim
    assert terminal_vals.shape[-1] == sde.dim if enforce_terminal_constraint else True

    SolverState = namedtuple(
        "SolverState", ["val", "scaled_stochastic_increment", "step_rng"]
    )
    init_state = SolverState(
        val=initial_vals,
        scaled_stochastic_increment=jnp.empty_like(initial_vals),
        step_rng=rng,
    )

    def euler_maruyama_step(state: SolverState, time: ArrayLike) -> tuple:
        """Euler-Maruyama step"""
        new_rng, _ = jax.random.split(state.step_rng)
        drift_step = sde.drift(state.val, time) * sde.dt
        brownian_step = jnp.sqrt(sde.dt) * jax.random.normal(
            new_rng, shape=state.val.shape
        )
        diffusion_step = batch_batch_multi(
            sde.diffusion(state.val, time), brownian_step
        )
        scaled_stochastic_increment = (
            -batch_multi(sde.inv_diffusion(state.val, time), brownian_step) / sde.dt
        )
        new_val = state.val + drift_step + diffusion_step
        new_state = SolverState(
            val=new_val,
            scaled_stochastic_increment=scaled_stochastic_increment,
            step_rng=new_rng,
        )
        return new_state, (state.val, state.scaled_stochastic_increment, state.step_rng)

    _, (trajectories, scaled_stochastic_increments, step_rngs) = jax.lax.scan(
        euler_maruyama_step, init=init_state, xs=(sde.ts[:-1])
    )

    if enforce_terminal_constraint:
        trajectories = trajectories.at[-1].set(terminal_vals)
    return {
        "trajectories": jnp.swapaxes(trajectories, 0, 1),
        "scaled_stochastic_increments": jnp.swapaxes(
            scaled_stochastic_increments, 0, 1
        ),
        "step_rngs": step_rngs,
    }


@jax.vmap
def weighted_norm_square(x: jax.Array, weight: jax.Array) -> jax.Array:
    assert x.shape[0] == weight.shape[0]
    Wx = jnp.dot(weight, x)
    xWx = jnp.dot(x, Wx)
    return xWx


@jax.vmap
def normal_norm_square(x: jax.Array, weight: jax.Array) -> jax.Array:
    return jnp.square(x)


def Q_kernel(distance: jax.Array, alpha: float, sigma: float) -> jax.Array:
    return (
        0.5
        * (alpha**2)
        * (sigma**2)
        * jnp.pi
        * jnp.exp(-0.5 * jnp.sum(jnp.square(distance), axis=-1) / (sigma**2))
    )


def eval_Q(
    x: jax.Array, alpha: float, sigma: float
) -> jax.Array:  # evaluate for a single point
    x_coords = x.reshape(-1, 2)
    dim = x_coords.shape[0]
    relative_distance = x_coords[:, jnp.newaxis, :] - x_coords[jnp.newaxis, :, :]
    kernel = Q_kernel(relative_distance, alpha, sigma)
    Q = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
    Q = Q.reshape(2 * dim, 2 * dim)
    return Q


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
