from functools import partial
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
from flax import linen as nn
import optax
from clu import metrics
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

### Dimension helpers
def flatten_batch(x: jnp.ndarray):
    assert len(x.shape) >= 2
    return x.reshape(-1, x.shape[-1])

def unsqueeze(x: jnp.ndarray, axis: int):
    return jnp.expand_dims(x, axis=axis)

def squeeze(x: jnp.ndarray, axis: int):
    return jnp.squeeze(x, axis=axis)

def sb_multi(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """ Perform matrix-vector multiplication for a single matrix/vector and each matrix/vector in a batch.
    """
    return jax.vmap(lambda x, y: jnp.dot(x, y), in_axes=(None, 0), out_axes=0)(a, b)

def bb_multi(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """ Perform matrix_vector multiplication for each pair of matrix/vector and matrix/vector in two batches.
    """
    return jax.vmap(lambda x, y: jnp.dot(x, y), in_axes=(0, 0), out_axes=0)(a, b)

### Network helpers
@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    batch_stats: any
    metrics: Metrics

def create_train_state(module: nn.Module, rng: jnp.ndarray, learning_rate: float, input_shapes: list) -> TrainState:
    init_inputs = [jnp.zeros(shape=shape) for shape in input_shapes]
    variables = module.init(rng, *init_inputs, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats'] if 'batch_stats' in variables else {}

    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        metrics=Metrics.empty()
    )

@jax.jit
def eval_score(state: TrainState, x: jnp.ndarray, t: float) -> jnp.ndarray:
    assert len(x.shape) == 2                                        # (B, d)                                                       
    t = jnp.tile(t, reps=(x.shape[0], 1))                           # (B, 1)
    return state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, t, train=False)

def get_iterable_dataset(generator: callable, dtype: tf.DType, shape: tuple):
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(tf.TensorSpec(shape=shape, dtype=dtype)))
    iterable_dataset = iter(tfds.as_numpy(dataset))
    return iterable_dataset

### Plotting helpers
def plot_est_vector_field(X_est_state: TrainState,
                          xs: jnp.ndarray,
                          ts: jnp.ndarray,
                          suptitle: str,
                          **kwargs):
    xm, ym = jnp.meshgrid(xs, xs)
    x = jnp.stack([xm.flatten(), ym.flatten()], axis=-1)

    fig, ax = plt.subplots(1, len(ts), figsize=(4*len(ts), 4))
    for i, t in enumerate(ts):
        vector_field = eval_score(X_est_state, x, t)
        u = vector_field[:, 0].reshape(xm.shape)
        v = vector_field[:, 1].reshape(xm.shape)
        ax[i].quiver(xm, ym, u, v, scale=kwargs['scale'])
        ax[i].set_title(f"t = {t:.1f}")
    fig.suptitle(suptitle)
    plt.show()

def plot_vector_field(X: callable, 
                      xs: jnp.ndarray,
                      ts: jnp.ndarray,
                      suptitle: str,
                      **kwargs):
    xm, ym = jnp.meshgrid(xs, xs)
    x = jnp.stack([xm.flatten(), ym.flatten()], axis=-1)
    fig, ax = plt.subplots(1, len(ts), figsize=(4*len(ts), 4))
    X = partial(X, **kwargs)
    for i, t in enumerate(ts):
        vector_field = jax.vmap(X, in_axes=(0, None))(x, t)
        u = vector_field[:, 0].reshape(xm.shape)
        v = vector_field[:, 1].reshape(xm.shape)
        ax[i].quiver(xm, ym, u, v, scale=kwargs['scale'])
        ax[i].set_title(f"t = {t:.1f}")
    fig.suptitle(suptitle)
    plt.show()