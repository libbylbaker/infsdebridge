import jax
import jax.numpy as jnp
import pytest

from sdebridge.networks.score_unet import ScoreUNet


@pytest.fixture
def x():
    return jnp.ones((8, 16), dtype=jnp.complex64)


@pytest.fixture
def t():
    return jnp.ones((8, 1), dtype=jnp.float32)


def test_score_unet(x, t):
    key = jax.random.PRNGKey(0)
    net = ScoreUNet(
        output_dim=32,
        time_embedding_dim=32,
        init_embedding_dim=32,
        act_fn="elu",
        encoder_layer_dims=[16, 8, 4],
        decoder_layer_dims=[4, 8, 16],
        batchnorm=True,
    )
    variables = net.init(key, x_complex=x, t=t, train=False)
    params, batch_stats = variables["params"], variables["batch_stats"]
    score, updates = net.apply(
        {"params": params, "batch_stats": batch_stats},
        x_complex=x,
        t=t,
        train=True,
        mutable=["batch_stats"],
    )
    assert score.shape == (8, 32)
