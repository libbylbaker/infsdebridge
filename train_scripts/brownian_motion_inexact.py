import functools

import jax
import jax.numpy as jnp
import orbax
from flax.training import orbax_utils

from sdebridge import diffusion_bridge as db
from sdebridge import sdes
from sdebridge.data_processing import sample_ellipse
from sdebridge.networks.score_unet import ScoreUNet
from sdebridge.utils import score_fn

# Using landmarks

save_path = "/home/gefan/Projects/sdebridge/sdebridge/ckpts/bm/32_landmarks_inexact"

sde_config = {
    "T": 1.0,
    "Nt": 100,
    "dim": 2,
    "N": 32,
    "sigma": 1.0,
}

bm_sde = sdes.brownian_sde(**sde_config)

network = {
    "output_dim": bm_sde.dim * bm_sde.n_bases,
    "time_embedding_dim": 32,
    "init_embedding_dim": 32,
    "act_fn": "silu",
    "encoder_layer_dims": [64, 32, 16, 8],
    "decoder_layer_dims": [8, 16, 32, 64],
    "batchnorm": True,
}

training = {
    "batch_size": 100,
    "load_size": 5000,
    "num_epochs": 200,
    "learning_rate": 1e-2,
    "warmup_steps": 1000,
}

key = jax.random.PRNGKey(2)
train_key, val_key = jax.random.split(key, 2)

neural_net = ScoreUNet


def sample_multiple_circles(key, num_circs, num_pts, min_radius=0.7, max_radius=1.0, centre=jnp.asarray([0.0, 0.0])):
    radius = jax.random.uniform(key, shape=num_circs, minval=min_radius, maxval=max_radius)
    ellipse_fn = functools.partial(sample_ellipse, num_pts=num_pts, shifts=centre)
    circs = jax.vmap(ellipse_fn, [0, None])(radius)
    return circs


target_sampler = functools.partial(sample_multiple_circles, num_pts=sde_config["N"])

score_state_p = db.learn_p_score(
    bm_sde, target_sampler, train_key, aux_dim=1, **training, net=neural_net, network_params=network
)

score_p_ckpt = {
    "state": score_state_p,
    "training_config": training,
    "network_config": network,
    "sde_config": sde_config,
}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(score_p_ckpt)
orbax_checkpointer.save(save_path, score_p_ckpt, save_args=save_args)
print("saved score_p_ckpt")
