import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time

import jax
import jax.numpy as jnp
import orbax
from flax.training import orbax_utils

from sdebridge import diffusion_bridge as db
from sdebridge import sdes, utils
from sdebridge.data_processing import sample_ellipse
from sdebridge.networks.score_unet import ScoreUNet

parser = argparse.ArgumentParser()
parser.add_argument("--n_bases", type=int, default=64)


def run(n_bases):
    save_path = f"./train_scripts/ckpts/bm/{n_bases}_landmarks_exact"
    sde_config = {
        "T": 1.0,
        "Nt": 100,
        "dim": 2,
        "n_bases": n_bases,
        "sigma": 0.1,
    }

    bm_sde = sdes.brownian_sde(**sde_config)

    network = {
        "output_dim": bm_sde.dim * bm_sde.n_bases,
        "time_embedding_dim": bm_sde.dim * bm_sde.n_bases * 4,
        "init_embedding_dim": bm_sde.dim * bm_sde.n_bases * 4,
        "act_fn": "silu",
        "encoder_layer_dims": [
            bm_sde.dim * bm_sde.n_bases * 8,
            bm_sde.dim * bm_sde.n_bases * 4,
            bm_sde.dim * bm_sde.n_bases * 2,
            bm_sde.dim * bm_sde.n_bases * 1,
        ],
        "decoder_layer_dims": [
            bm_sde.dim * bm_sde.n_bases * 1,
            bm_sde.dim * bm_sde.n_bases * 2,
            bm_sde.dim * bm_sde.n_bases * 4,
            bm_sde.dim * bm_sde.n_bases * 8,
        ],
        "batchnorm": True,
    }

    training = {
        "batch_size": 50,
        "load_size": 2000,
        "num_epochs": 300,
        "learning_rate": 2e-3,
        "warmup_steps": 500,
    }

    key = jax.random.PRNGKey(2)

    neural_net = ScoreUNet
    target = sample_ellipse(n_bases)
    target = jnp.expand_dims(target, axis=0)

    def target_sampler(key, num_batches):
        initial_vals = jnp.tile(target, reps=(num_batches, 1, 1, 1))
        return initial_vals

    train_key = jax.random.split(key, 2)[0]
    start_time = time.time()
    score_state_p = db.learn_p_score(
        bm_sde, target_sampler, train_key, aux_dim=1, **training, net=neural_net, network_params=network
    )
    end_time = time.time()
    t = end_time - start_time

    score_p_ckpt = {
        "state": score_state_p,
        "training_config": training,
        "network_config": network,
        "sde_config": sde_config,
        "time": t,
    }

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    save_args = orbax_utils.save_args_from_target(score_p_ckpt)
    orbax_checkpointer.save(os.path.abspath(save_path), score_p_ckpt, save_args=save_args, force=True)
    print("saved score_p_ckpt")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(f"./ckpts/bm/{args.n_bases}_fourier_exact"):
        os.makedirs(f"./ckpts/bm/{args.n_bases}_fourier_exact")

    run(args.n_bases)
