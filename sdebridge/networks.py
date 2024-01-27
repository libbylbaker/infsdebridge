from typing import Any

from flax import linen as nn

# from .setup import *

import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Sequence
import math
import jax
from functools import partial

dense = partial(nn.Dense,
                )

def scaled_dot_product(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    dim_k = q.shape[-1]
    attn_scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(dim_k)
    attn = nn.softmax(attn_scores, axis=-1)
    values = jnp.matmul(attn, v)
    return values


def get_time_step_embedding(
    time_steps: ArrayLike,
    embedding_dim: int,
    max_period: int = 10000,
    scaling_factor: float = 2000.0,
) -> jnp.ndarray:
    def encode_scalar(t: ArrayLike) -> jnp.ndarray:
        k = embedding_dim // 2
        emb = jnp.log(max_period) / (k - 1)
        emb = jnp.exp(jnp.arange(k, dtype=jnp.float32) * -emb)
        emb = scaling_factor * t * emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

    if len(time_steps.shape) == 0 or time_steps.shape[0] == 1:
        return encode_scalar(time_steps)
    else:
        return jax.vmap(encode_scalar)(time_steps)


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


class Upsample(nn.Module):
    output_dim: int
    act_fn: str
    batchnorm: bool = False
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(
        self, x1: jnp.ndarray, x2: jnp.ndarray, t_emb: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        x_in = jnp.concatenate([x1, x2], axis=-1)
        input_dim = x_in.shape[-1]
        x = nn.Dense(input_dim, kernel_init=nn.initializers.xavier_normal())(x_in)
        x = nn.Dropout(rate=self.dropout_prob)(x, deterministic=not train)
        scale, shift = TimeEmbeddingMLP(input_dim, self.act_fn)(t_emb)
        x = x * (1.0 + scale) + shift
        x = x_in + get_act_fn(self.act_fn)(x)
        x = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.Dropout(rate=self.dropout_prob)(x, deterministic=not train)
        x = get_act_fn(self.act_fn)(x)
        if self.batchnorm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x


class Downsample(nn.Module):
    output_dim: int
    act_fn: str
    batchnorm: bool = False
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        x_in = x
        input_dim = x_in.shape[-1]
        x = nn.Dense(input_dim, kernel_init=nn.initializers.xavier_normal())(x_in)
        x = nn.Dropout(rate=self.dropout_prob)(x, deterministic=not train)
        scale, shift = TimeEmbeddingMLP(input_dim, self.act_fn)(t_emb)
        x = x * (1.0 + scale) + shift
        x = x_in + get_act_fn(self.act_fn)(x)
        x = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.Dropout(rate=self.dropout_prob)(x, deterministic=not train)
        x = get_act_fn(self.act_fn)(x)
        if self.batchnorm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x


class TimeEmbeddingMLP(nn.Module):
    output_dim: int
    act_fn: str

    @nn.compact
    def __call__(self, t_emb: jnp.ndarray) -> jnp.ndarray:
        scale_shift = nn.Dense(
            2 * self.output_dim, kernel_init=nn.initializers.xavier_uniform()
        )(t_emb)
        scale, shift = jnp.array_split(scale_shift, 2, axis=-1)
        return scale, shift


class ScoreUNet(nn.Module):
    output_dim: int
    time_embedding_dim: int
    encoding_dim: int
    act_fn: str
    encoder_layer_dims: Sequence[int]
    decoder_layer_dims: Sequence[int]
    batchnorm: bool = False
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        t_emb = get_time_step_embedding(t, self.time_embedding_dim)
        x = nn.Dense(
            self.time_embedding_dim, kernel_init=nn.initializers.xavier_uniform()
        )(x)

        # downsample
        downs = []
        for dim in self.encoder_layer_dims:
            x = Downsample(
                output_dim=dim,
                act_fn=self.act_fn,
                batchnorm=self.batchnorm,
                dropout_prob=self.dropout_prob,
            )(x, t_emb, train)
            downs.append(x)

        # bottleneck
        x_out = nn.Dense(
            self.encoding_dim, kernel_init=nn.initializers.xavier_uniform()
        )(x)
        x_out = nn.Dropout(rate=self.dropout_prob)(x, deterministic=not train)
        x = x + get_act_fn(self.act_fn)(x_out)

        # upsample
        for dim, down in zip(self.decoder_layer_dims, downs[::-1]):
            x = Upsample(
                output_dim=dim,
                act_fn=self.act_fn,
                batchnorm=self.batchnorm,
                dropout_prob=self.dropout_prob,
            )(x, down, t_emb, train)

        # out
        score = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(
            x
        )
        return score


if __name__ == "__main__":
    x = jnp.ones((4, 2, 16), dtype=jnp.complex64)
    t = jnp.ones((4, 2, 1), dtype=jnp.float32)
    net = ScoreUNet(
        output_dim=32,
        time_embedding_dim=32,
        encoding_dim=8,
        act_fn="elu",
        encoder_layer_dims=[16, 8],
        decoder_layer_dims=[8, 16],
        batchnorm=True,
        dropout_prob=0.1,
    )
    key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
    variable = net.init({"params": key1, "dropout": key2}, x=x, t=t, train=True)
    params = variable["params"]
    batch_stats = variable["batch_stats"] if "batch_stats" in variable else {}
    score = net.apply(
        {"params": params, "batch_stats": batch_stats},
        x=x,
        t=t,
        train=False,
        mutable=False,
    )
    print(score.shape)
    print(score.dtype)
