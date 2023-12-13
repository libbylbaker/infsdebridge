import jax.numpy as jnp
from flax import linen as nn

def get_timestep_embedding(timesteps: jnp.ndarray, embedding_dim: int, max_period: int=10000, scaling_factor: float=100.0) -> jnp.ndarray:
    assert len(timesteps.shape) == 2 and timesteps.shape[-1] == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(max_period) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = scaling_factor * timesteps * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def get_activation(name: str) -> nn.activation:
    if name == 'relu':
        return nn.relu
    elif name == 'leaky_relu':
        return nn.leaky_relu
    elif name == 'tanh':
        return nn.tanh
    elif name == 'sigmoid':
        return nn.sigmoid
    else:
        raise ValueError(f'Activation {name} not recognized.')

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
    embedding_dim: int
    act: str
    encoder_layer_dims: list
    decoder_layer_dims: list
    using_batchnorm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        assert len(x.shape) == len(t.shape) == 2
        t = get_timestep_embedding(t, self.time_embedding_dim)
        t = MLP(out_dim=self.embedding_dim,
                act=self.act,
                layer_dims=self.encoder_layer_dims,
                apply_act_at_output=False,
                using_batchnorm=self.using_batchnorm)(t, train)
        x = MLP(out_dim=self.embedding_dim,
                act=self.act,
                layer_dims=self.encoder_layer_dims,
                apply_act_at_output=False,
                using_batchnorm=self.using_batchnorm)(x, train)
        assert t.shape == x.shape == (x.shape[0], self.embedding_dim)
        xt = jnp.concatenate([x, t], axis=-1)
        score = MLP(out_dim=self.out_dim,
                    act=self.act,
                    layer_dims=self.decoder_layer_dims,
                    apply_act_at_output=False,
                    using_batchnorm=self.using_batchnorm)(xt, train)
        return score