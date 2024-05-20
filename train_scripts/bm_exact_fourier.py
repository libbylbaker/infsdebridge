import os.path

import jax
import jax.numpy as jnp
import orbax
from flax.training import orbax_utils

from sdebridge import diffusion_bridge as db
from sdebridge import sdes, utils
from sdebridge.data_processing import sample_ellipse
from sdebridge.networks.score_unet import ScoreUNet


def run(n_bases):
    save_path = f"./ckpts/bm/{n_bases}_fourier_exact"
    sde_config = {
        "T": 1.0,
        "Nt": 100,
        "dim": 2,
        "n_bases": n_bases,
        "sigma": 1.0,
    }

    bm_sde = sdes.brownian_sde(**sde_config)

    network = {
        "output_dim": bm_sde.dim * bm_sde.n_bases * 2,
        "time_embedding_dim": 32,
        "init_embedding_dim": 32,
        "act_fn": "silu",
        "encoder_layer_dims": [32, 16, 8],
        "decoder_layer_dims": [8, 16, 32],
        "batchnorm": True,
    }

    training = {
        "batch_size": 100,
        "load_size": 5000,
        "num_epochs": 100,
        "learning_rate": 2e-3,
        "warmup_steps": 0,
    }

    key = jax.random.PRNGKey(2)

    neural_net = ScoreUNet
    target = sample_ellipse(100)
    target = utils.fourier_coefficients(target, n_bases)

    def target_sampler(key, num_batches):
        initial_vals = jnp.tile(target, reps=(num_batches, 1, 1, 1))
        return initial_vals

    train_key = jax.random.split(key, 2)[0]
    score_state_p = db.learn_p_score(
        bm_sde, target_sampler, train_key, aux_dim=2, **training, net=neural_net, network_params=network
    )

    score_p_ckpt = {
        "state": score_state_p,
        "training_config": training,
        "network_config": network,
        "sde_config": sde_config,
    }

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    save_args = orbax_utils.save_args_from_target(score_p_ckpt)
    orbax_checkpointer.save(os.path.abspath(save_path), score_p_ckpt, save_args=save_args, force=True)
    print("saved score_p_ckpt")


run(4)
