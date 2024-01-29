import math
from functools import partial
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def get_time_step_embedding(
    t: float,
    embedding_dim: int,
    max_period: float = 128.0,
    scaling: float = 100.0,
) -> jnp.ndarray:
    """ Embed scalar time steps into a vector of size `embedding_dim`"""
    emb = jnp.empty((embedding_dim, ), dtype=jnp.float32)
    div_term = jnp.exp(jnp.arange(0, embedding_dim, 2, dtype=jnp.float32) * (-math.log(max_period) / embedding_dim))
    emb = emb.at[0::2].set(jnp.sin(scaling * t * div_term))
    emb = emb.at[1::2].set(jnp.cos(scaling * t * div_term))
    return emb

def get_act_fn(name: str) -> nn.activation:
    if name == "relu":
        return nn.relu
    elif name == "leaky_relu":
        return nn.leaky_relu
    elif name == "elu":
        return nn.elu
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    elif name == "tanh":
        return nn.tanh
    elif name == "sigmoid":
        return nn.sigmoid
    elif name == "none":
        return lambda x: x
    else:
        raise ValueError(f"Activation {name} not recognized.")

class TimeEmbeddingMLP(nn.Module):
    output_dim: int
    act_fn: str

    @nn.compact
    def __call__(self, t_emb: jnp.ndarray) -> jnp.ndarray:
        scale_shift = nn.Dense(
            2 * self.output_dim,
            kernel_init=nn.initializers.xavier_normal()
        )(t_emb)
        scale, shift = jnp.array_split(scale_shift, 2, axis=-1)
        return scale, shift
    
class InputDense(nn.Module):
    output_dims: int
    act_fn: str
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x_complex: jnp.ndarray) -> jnp.ndarray:
        x_real, x_complex = jnp.real(x_complex), jnp.imag(x_complex)
        x = jnp.concatenate([x_real, x_complex], axis=-1)
        x = nn.Dense(self.output_dims,
                     kernel_init=self.kernel_init)(x)
        x = get_act_fn(self.act_fn)(x)
        return x
    
class Dense(nn.Module):
    output_dims: int
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.output_dims,
                     kernel_init=self.kernel_init)(x)
        return x

class Downsample(nn.Module):
    output_dim: int
    act_fn: str
    batchnorm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        x_res = Dense(self.output_dim)(x)
        x = Dense(self.output_dim)(x)

        scale, shift = TimeEmbeddingMLP(self.output_dim,
                                        self.act_fn)(t_emb)
        x = x * (1.0 + scale) + shift
        x = get_act_fn(self.act_fn)(x)
        x = x + x_res
        if self.batchnorm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x
    
class Upsample(nn.Module):
    output_dim: int
    act_fn: str
    batchnorm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, x_skip: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = jnp.concatenate([x, x_skip], axis=-1)
    
        x_res = Dense(self.output_dim)(x)
        x = Dense(self.output_dim)(x)

        scale, shift = TimeEmbeddingMLP(self.output_dim, 
                                        self.act_fn)(t_emb)
        x = x * (1.0 + scale) + shift
        x = get_act_fn(self.act_fn)(x)
        x = x + x_res
        if self.batchnorm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x

class ScoreUNet(nn.Module):
    output_dim: int
    time_embedding_dim: int
    init_embedding_dim: int
    act_fn: str
    encoder_layer_dims: Sequence[int]
    decoder_layer_dims: Sequence[int]
    batchnorm: bool = True

    @nn.compact
    def __call__(self, x_complex: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        assert self.encoder_layer_dims[-1] == self.decoder_layer_dims[0], "Bottleneck dim does not match"
        t_emb = jax.vmap(
            partial(get_time_step_embedding,
                    embedding_dim=self.time_embedding_dim)
            )(t)
        x = InputDense(self.init_embedding_dim,
                       self.act_fn)(x_complex)

        # downsample
        downs = []
        for dim in self.encoder_layer_dims:
            x = Downsample(
                output_dim=dim,
                act_fn=self.act_fn,
                batchnorm=self.batchnorm,
            )(x, t_emb, train)
            downs.append(x)

        # bottleneck
        bottleneck_dim = self.encoder_layer_dims[-1]
        x = Dense(bottleneck_dim)(x)
        x = get_act_fn(self.act_fn)(x)

        # upsample
        for dim, x_skip in zip(self.decoder_layer_dims, downs[::-1]):
            x = Upsample(
                output_dim=dim,
                act_fn=self.act_fn,
                batchnorm=self.batchnorm,
            )(x, x_skip, t_emb, train)

        # out
        score = Dense(self.output_dim)(x)
        return score


if __name__ == "__main__":
    x = jnp.ones((8, 16), dtype=jnp.complex64)
    t = jnp.ones((8, 1), dtype=jnp.float32)
    net = ScoreUNet(
        output_dim=32,
        time_embedding_dim=32,
        init_embedding_dim=32,
        act_fn="elu",
        encoder_layer_dims=[16, 8, 4],
        decoder_layer_dims=[4, 8, 16],
        batchnorm=True,
    )
    key = jax.random.PRNGKey(0)
    variables = net.init(key, x_complex=x, t=t, train=False)
    params, batch_stats = variables["params"], variables["batch_stats"]
    score, updates = net.apply(
        {"params": params, "batch_stats": batch_stats},
        x_complex=x,
        t=t,
        train=True,
        mutable=["batch_stats"]
    )
    print(score.shape)
