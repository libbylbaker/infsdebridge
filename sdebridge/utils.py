from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax import struct
from flax.training import train_state


### Dimension helpers
def flatten_batch(x: jax.Array):
    assert len(x.shape) >= 2
    return x.reshape(-1, x.shape[-1])


def unsqueeze(x: jax.Array, axis: int):
    return jnp.expand_dims(x, axis=axis)


### Network helpers


class TrainState(train_state.TrainState):
    key: jax.Array
    batch_stats: any


def create_optimizer(learning_rate: float, warmup_steps: int, decay_steps: int):
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=0.01 * learning_rate,
    )
    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(lr_scheduler),
    )
    return optimizer


def create_train_state(
    model: nn.Module,
    rng_key: jax.Array,
    input_shapes: Sequence[tuple],
    learning_rate: float,
    warmup_steps: int = 500,
    decay_steps: int = 2000,
) -> TrainState:
    rng_key, params_init_rng_key, dropout_init_rng_key = jax.random.split(rng_key, 3)
    init_inputs = [jnp.zeros(shape=shape) for shape in input_shapes]
    variables = model.init(
        {"params": params_init_rng_key, "dropout": dropout_init_rng_key},
        *init_inputs,
        train=True,
    )
    params = variables["params"]
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}

    optimizer = create_optimizer(
        learning_rate=learning_rate, warmup_steps=warmup_steps, decay_steps=decay_steps
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        key=rng_key,
        tx=optimizer,
        batch_stats=batch_stats,
    )

    # print(parameter_overview.get_parameter_overview(params))
    return state


@jax.jit
def eval_score(state: TrainState, val: jax.Array, time: jnp.ndarray) -> jax.Array:
    assert len(val.shape) == 1
    time = jnp.array(time)
    val_real, val_imag = val.real, val.imag
    score_real, score_imag = state.apply_fn(
        {"params": state.params},
        x_real=val_real,
        x_imag=val_imag,
        t=time,
        train=False,
    )
    return score_real + 1j * score_imag


def score_fn(state: TrainState) -> Callable:
    @jax.jit
    def score(val, time):
        result = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x=val[None],
            t=jnp.asarray([time]),
            train=False,
        )
        return result

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


@jax.jit
@jax.vmap
def weighted_norm_square(x: jax.Array, covariance: jax.Array) -> jax.Array:
    assert x.shape[0] == covariance.shape[0]
    Wx = (covariance @ x).flatten()
    xWx = jnp.dot(x.flatten(), Wx)
    return xWx
