from typing import Any

from flax import linen as nn

from .setup import *

# import numpy as np
# import jax.numpy as jnp
# from typing import Sequence, Callable
# import math
# import jax
# from functools import partial
# from clu import parameter_overview

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

class ComplexDense(nn.Module):
    features: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.xavier_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x_real: jnp.ndarray, x_imag: jnp.ndarray) -> jnp.ndarray:
        x_real = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x_real)
        x_imag = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x_imag)
        return x_real, x_imag

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

class Downsample(nn.Module):
    output_dim: int
    act_fn: str

    @nn.compact
    def __call__(self, x_real: jnp.ndarray, x_imag: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        x_real_res, x_imag_res = ComplexDense(self.output_dim)(x_real, x_imag)
        x_real, x_imag = ComplexDense(self.output_dim)(x_real, x_imag)

        scale, shift = TimeEmbeddingMLP(self.output_dim,
                                        self.act_fn)(t_emb)
        x_real = x_real * (1.0 + scale) + shift
        x_imag = x_imag * (1.0 + scale) + shift
        x_real = get_act_fn(self.act_fn)(x_real)
        x_imag = get_act_fn(self.act_fn)(x_imag)

        x_real = x_real + x_real_res
        x_imag = x_imag + x_imag_res

        x_real = nn.LayerNorm()(x_real)
        x_imag = nn.LayerNorm()(x_imag)
        return x_real, x_imag
    
class Upsample(nn.Module):
    output_dim: int
    act_fn: str

    @nn.compact
    def __call__(
        self, x_real: jnp.ndarray, x_imag: jnp.ndarray, x_real_skip: jnp.ndarray, x_imag_skip: jnp.ndarray, t_emb: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        x_real = jnp.concatenate([x_real, x_real_skip], axis=-1)
        x_imag = jnp.concatenate([x_imag, x_imag_skip], axis=-1)
        
        x_real_res, x_imag_res = ComplexDense(self.output_dim)(x_real, x_imag)
        x_real, x_imag = ComplexDense(self.output_dim)(x_real, x_imag)

        scale, shift = TimeEmbeddingMLP(self.output_dim, 
                                        self.act_fn)(t_emb)
        x_real = x_real * (1.0 + scale) + shift
        x_imag = x_imag * (1.0 + scale) + shift
        x_real = get_act_fn(self.act_fn)(x_real)
        x_imag = get_act_fn(self.act_fn)(x_imag)

        x_real = x_real + x_real_res
        x_imag = x_imag + x_imag_res

        x_real = nn.LayerNorm()(x_real)
        x_imag = nn.LayerNorm()(x_imag)
        return x_real, x_imag

class ScoreUNet(nn.Module):
    output_dim: int
    time_embedding_dim: int
    bottleneck_dim: int
    act_fn: str
    encoder_layer_dims: Sequence[int]
    decoder_layer_dims: Sequence[int]

    @nn.compact
    def __call__(self, x_real: jnp.ndarray, x_imag: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        t_emb = jax.vmap(
            partial(get_time_step_embedding,
                    embedding_dim=self.time_embedding_dim)
            )(t)
        x_real, x_imag = ComplexDense(self.time_embedding_dim)(x_real, x_imag)

        # downsample
        x_reals, x_imags = [], []
        for dim in self.encoder_layer_dims:
            x_real, x_imag = Downsample(
                output_dim=dim,
                act_fn=self.act_fn
            )(x_real, x_imag, t_emb, train)
            x_reals.append(x_real)
            x_imags.append(x_imag)

        # bottleneck
        x_real, x_imag = ComplexDense(self.bottleneck_dim)(x_real, x_imag)
        x_real = get_act_fn(self.act_fn)(x_real)
        x_imag = get_act_fn(self.act_fn)(x_imag)

        # upsample
        for dim, x_real_skip, x_imag_skip in zip(self.decoder_layer_dims[::-1], x_reals[::-1], x_imags[::-1]):
            x_real, x_imag = Upsample(
                output_dim=dim,
                act_fn=self.act_fn,
            )(x_real, x_imag, x_real_skip, x_imag_skip, t_emb, train)

        # out
        score_real, score_imag = ComplexDense(self.output_dim)(x_real, x_imag)
        return score_real, score_imag


if __name__ == "__main__":
    x = jnp.ones((8, 16), dtype=jnp.complex64)
    x_real, x_imag = x.real, x.imag
    t = jnp.ones((8, 1), dtype=jnp.float32)
    net = ScoreUNet(
        output_dim=16,
        time_embedding_dim=32,
        bottleneck_dim=8,
        act_fn="elu",
        encoder_layer_dims=[16, 8],
        decoder_layer_dims=[8, 16],
    )
    key = jax.random.PRNGKey(0)
    params = net.init(key, x_real=x_real, x_imag=x_imag, t=t, train=True)["params"]
    print(parameter_overview.get_parameter_overview(params))
    score_real, score_imag = net.apply(
        {"params": params},
        x_real=x_real,
        x_imag=x_imag,
        t=t,
        train=False,
    )
    print(score_real.shape)
    print(score_imag.shape)
    print(score_real.dtype)
    print(score_imag.dtype)
