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


def sparseify(x: jnp.ndarray, num_adjacent_nbs: int):
    x_copy = x.copy()
    for n in range(num_adjacent_nbs + 1, len(x) - num_adjacent_nbs):
        # Set the elements at the diagonal `n` away from the main diagonal to zero
        x_copy -= jnp.diagflat(jnp.diag(x_copy, k=n), k=n)
        x_copy -= jnp.diagflat(jnp.diag(x_copy, k=-n), k=-n)
    return x_copy


### Network helpers
@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    batch_stats: any
    metrics: Metrics


def create_train_state(
    module: nn.Module, rng_key: jax.Array, learning_rate: float, input_shapes: list
) -> TrainState:
    init_inputs = [jnp.zeros(shape=shape) for shape in input_shapes]
    variables = module.init(rng_key, *init_inputs, train=False)
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
def eval_scores(state: TrainState, vals: jax.Array, time: ArrayLike) -> jax.Array:
    assert len(vals.shape) == 2
    times = jnp.tile(jnp.array(time), (vals.shape[0], 1))
    scores = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        x=vals,
        t=times,
        train=False,
    )
    return scores


@jax.jit
def eval_score(state: TrainState, val: jax.Array, time: ArrayLike) -> jax.Array:
    assert len(val.shape) == 1
    time = jnp.array(time)
    score = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        x=val,
        t=time,
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


@jax.vmap
def weighted_norm_square(x: jax.Array, weight: jax.Array) -> jax.Array:
    assert x.shape[0] == weight.shape[0]
    Wx = jnp.dot(weight, x)
    xWx = jnp.dot(x, Wx)
    return xWx


def Q_kernel(distance: jax.Array, alpha: float, sigma: float) -> jax.Array:
    return (alpha**2) * jnp.exp(
        -0.5 * jnp.sum(jnp.square(distance), axis=-1) / (sigma**2)
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


### Data helpers
def sample_circle(num_points: int, scale: float, shifts: jax.Array) -> jax.Array:
    theta = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False)
    x = jnp.cos(theta)
    y = jnp.sin(theta)
    return (scale * jnp.stack([x, y], axis=1) + shifts).flatten()


def sample_ellipse(
    num_points: int, scale: float, shifts: jax.Array, a: float, b: float
) -> jax.Array:
    theta = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False)
    x = a * jnp.cos(theta)
    y = b * jnp.sin(theta)
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
