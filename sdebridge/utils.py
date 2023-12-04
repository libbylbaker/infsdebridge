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
def plot_2d_vector_field(X: callable, 
                         X_ref: callable,
                         xs: jnp.ndarray,
                         ts: jnp.ndarray,
                         suptitle: str,
                         scale: float=None,
                         **kwargs):
    xm, ym = jnp.meshgrid(xs, xs)
    x = jnp.stack([xm.flatten(), ym.flatten()], axis=-1)
    fig, ax = plt.subplots(1, len(ts), figsize=(4*len(ts), 4))
    X_ref = partial(X_ref, **kwargs) if X_ref is not None else None
    for i, t in enumerate(ts):
        vector_field = X(x, t) if X is not None else None
        vector_field_ref = X_ref(x, t) if X_ref is not None else None
        if vector_field is not None:
            u = vector_field[:, 0].reshape(xm.shape)
            v = vector_field[:, 1].reshape(xm.shape)
            ax[i].quiver(xm, ym, u, v, color='b', scale=scale)

        if vector_field_ref is not None:
            u_ref = vector_field_ref[:, 0].reshape(xm.shape)
            v_ref = vector_field_ref[:, 1].reshape(xm.shape)
            ax[i].quiver(xm, ym, u_ref, v_ref, color='r', scale=scale)
        
        ax[i].set_title(f"t = {t:.1f}")
    fig.suptitle(suptitle)
    plt.show() 

def plot_2d_trajectories(trajectories: jnp.ndarray, title: str, **kwargs):
    colormap = plt.cm.get_cmap('spring')
    num_trajectories = trajectories.shape[0]
    colors = [colormap(i) for i in jnp.linspace(0, 1, num_trajectories)]
    for i in range(num_trajectories):
        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], color=colors[i], zorder=1, alpha=0.5, **kwargs)
        plt.scatter(trajectories[i, 1, 0], trajectories[i, 1, 1], color='b', marker='o', edgecolors='k', zorder=2)
        plt.scatter(trajectories[i, -2, 0], trajectories[i, -2, 1], color='c', marker='D', edgecolors='k', zorder=2)
    plt.title(title)

def plot_trajectories(trajectories: jnp.ndarray, title: str, **kwargs):
    colormap = plt.cm.get_cmap('spring')
    assert len(trajectories.shape) == 3
    num_trajectories = trajectories.shape[0]
    dim = trajectories.shape[2]
    colors = [colormap(i) for i in jnp.linspace(0, 1, num_trajectories)]
    for i in range(num_trajectories):
        for j in range(dim//2):
            plt.plot(trajectories[i, :, 2*j], trajectories[i, :, 2*j+1], color=colors[i], zorder=1, alpha=0.2, **kwargs)
            plt.scatter(trajectories[i, 0, 2*j], trajectories[i, 0, 2*j+1], color='b', marker='o', edgecolors='k', zorder=2)
            plt.scatter(trajectories[i, -1, 2*j], trajectories[i, -1, 2*j+1], color=colors[i], marker='D', edgecolors='k', zorder=2)
    plt.title(title)