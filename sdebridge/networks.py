import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.typing import ArrayLike


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


def get_activation(name: str) -> nn.activation:
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


class MLP(nn.Module):
    out_dim: int
    act: str
    layer_dims: list
    apply_act_at_output: bool
    using_batchnorm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        for dim in self.layer_dims:
            x = get_activation(self.act)(nn.Dense(dim)(x))
            if self.using_batchnorm:
                x = nn.BatchNorm(use_running_average=not train)(x)
        if self.apply_act_at_output:
            x = get_activation(self.act)(nn.Dense(self.out_dim)(x))
        else:
            x = nn.Dense(self.out_dim)(x)
        return x


class ScoreNet(nn.Module):
    out_dim: int
    time_embedding_dim: int
    encoding_dim: int
    act: str
    encoder_layer_dims: list
    decoder_layer_dims: list
    using_batchnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        t = get_time_step_embedding(t, self.time_embedding_dim)
        t = MLP(
            out_dim=self.encoding_dim,
            act=self.act,
            layer_dims=self.encoder_layer_dims,
            apply_act_at_output=False,
            using_batchnorm=self.using_batchnorm,
        )(t, train)
        x = MLP(
            out_dim=self.encoding_dim,
            act=self.act,
            layer_dims=self.encoder_layer_dims,
            apply_act_at_output=False,
            using_batchnorm=self.using_batchnorm,
        )(x, train)
        xt = jnp.concatenate([x, t], axis=-1)
        score = MLP(
            out_dim=self.out_dim,
            act=self.act,
            layer_dims=self.decoder_layer_dims,
            apply_act_at_output=False,
            using_batchnorm=self.using_batchnorm,
        )(xt, train)
        return score


if __name__ == "__main__":
    net = ScoreNet(
        out_dim=4,
        time_embedding_dim=32,
        encoding_dim=32,
        act="relu",
        encoder_layer_dims=[32, 32],
        decoder_layer_dims=[32, 32],
    )

    x = jnp.ones((4,))
    t = jnp.ones((1,))
    params = net.init(jax.random.PRNGKey(0), x, t, train=True)
    score = net.apply(params, x, t, train=True)
    print("x.shape: ", x.shape)
    print("t.shape: ", t.shape)
    print("score.shape: ", score.shape)
    print("--" * 10)

    x = jnp.ones((16, 4))
    t = jnp.ones((16, 1))
    score = net.apply(params, x, t, train=True)
    print("x.shape: ", x.shape)
    print("t.shape: ", t.shape)
    print("score.shape: ", score.shape)
    print("--" * 10)

    x = jnp.array([1.0, 1.0, 1.0, 1.0])
    t = jnp.array(1.0)
    score = net.apply(params, x, t, train=True)
    print("x.shape: ", x.shape)
    print("t.shape: ", t.shape)
    print("score.shape: ", score.shape)
    print("--" * 10)

    x = jnp.array([1.0, 1.0, 1.0, 1.0])
    t = jnp.array([1.0])
    score = net.apply(params, x, t, train=True)
    print("x.shape: ", x.shape)
    print("t.shape: ", t.shape)
    print("score.shape: ", score.shape)
    print("--" * 10)
