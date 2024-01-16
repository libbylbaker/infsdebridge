from flax import linen as nn

from .setup import *


def get_time_step_embedding(
    time_steps: ArrayLike,
    embedding_dim: int,
    max_period: int = 10000,
    scaling_factor: float = 100.0,
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
    elif name == "tanh":
        return nn.tanh
    elif name == "sigmoid":
        return nn.sigmoid
    else:
        raise ValueError(f"Activation {name} not recognized.")


def xavier_init(
    rng_key: jax.Array, shape: Sequence[int], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    bound = math.sqrt(6.0) / math.sqrt(shape[0] + shape[1])
    return random.uniform(rng_key, shape, dtype, minval=-bound, maxval=bound)


class MLP(nn.Module):
    output_dim: int
    act_fn: str
    layer_dims: Sequence[int]
    kernel_init: Callable = xavier_init
    batchnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        for dim in self.layer_dims:
            x = nn.Dense(dim, kernel_init=self.kernel_init)(x)
            x = get_act_fn(self.act_fn)(x)
            if self.batchnorm:
                x = nn.BatchNorm(use_running_average=not train)(x)

            x = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        return x


class ScoreNet(nn.Module):
    output_dim: int
    time_embedding_dim: int
    encoding_dim: int
    act_fn: str
    encoder_layer_dims: Sequence[int]
    decoder_layer_dims: Sequence[int]
    kernel_init: Callable = xavier_init
    batchnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        t = get_time_step_embedding(t, self.time_embedding_dim)
        t = MLP(
            output_dim=self.encoding_dim,
            act_fn=self.act_fn,
            layer_dims=self.encoder_layer_dims,
            kernel_init=self.kernel_init,
            batchnorm=self.batchnorm,
        )(t, train)
        x = MLP(
            output_dim=self.encoding_dim,
            act_fn=self.act_fn,
            layer_dims=self.encoder_layer_dims,
            kernel_init=self.kernel_init,
            batchnorm=self.batchnorm,
        )(x, train)
        xt = jnp.concatenate([x, t], axis=-1)
        score = MLP(
            output_dim=self.output_dim,
            act_fn=self.act_fn,
            layer_dims=self.decoder_layer_dims,
            kernel_init=self.kernel_init,
            batchnorm=self.batchnorm,
        )(xt, train)

        return score


if __name__ == "__main__":
    times = jnp.linspace(0, 1, 100)
    time_embeddings = get_time_step_embedding(times, 16)
    plt.imshow(time_embeddings.T, aspect="auto")
    plt.show()
