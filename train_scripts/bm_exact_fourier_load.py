import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint

from sdebridge import diffusion_bridge as db
from sdebridge import plotting, sdes, utils
from sdebridge.data_processing import sample_ellipse
from sdebridge.networks.score_unet import ScoreUNet


def load_ckpt(sde_config, network_config, training_config, save_path):
    key = jax.random.PRNGKey(2)
    score_net = ScoreUNet(**network_config)
    num_batches_per_epoch = int(training_config["load_size"] / training_config["batch_size"])

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    empty_state = utils.create_train_state(
        model=score_net,
        key=key,
        input_shapes=[
            (training_config["batch_size"], 2 * sde_config["n_bases"] * sde_config["dim"]),
            (training_config["batch_size"], 1),
        ],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        decay_steps=training_config["num_epochs"] * num_batches_per_epoch,
    )

    target = {
        "state": empty_state,
        "training_config": training_config,
        "network_config": network_config,
        "sde_config": sde_config,
        "time": 0.0,
    }
    return orbax_checkpointer.restore(save_path, item=target)


if __name__ == "__main__":
    # n_bases = 4

    def restore_for_bases(n_bases):
        save_path_end = f"./train_scripts/ckpts/bm/{n_bases}_fourier_exact"
        save_path = os.path.abspath(save_path_end)
        print(save_path)
        sde_config = {
            "T": 1.0,
            "Nt": 100,
            "dim": 2,
            "n_bases": n_bases,
            "sigma": 0.1,
        }

        bm_sde = sdes.brownian_sde(**sde_config)

        network = {
            "output_dim": 2 * bm_sde.dim * bm_sde.n_bases,
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
            "num_epochs": 100,
            "learning_rate": 2e-3,
            "warmup_steps": 100,
        }

        restored = load_ckpt(sde_config, network, training, save_path)
        state = restored["state"]
        return state, bm_sde, restored


basis_list = [32]
for n_bases in basis_list:
    state, bm_sde, restored = restore_for_bases(n_bases)

    target_pts = sample_ellipse(100)
    target = utils.fourier_coefficients(target_pts, n_bases)

    def forward_score(t0, x0, t, x):
        x0 = jnp.asarray(x0)
        x = jnp.asarray(x)
        return -(x - x0) / (t - t0)

    def error_forward(ts, true_score, trained_score, target, y):
        """mean squared error between true and trained score"""
        true = jax.vmap(true_score, in_axes=(None, None, 0, None))(0, target, ts, y)
        trained = jax.vmap(trained_score, in_axes=(None, 0))(y, ts)
        true_landmark = utils.inverse_fourier(true, 100)
        trained_landmark = utils.inverse_fourier(trained, 100)
        return jnp.sqrt(jnp.mean((true_landmark - trained_landmark) ** 2))

    ts = jnp.linspace(0, bm_sde.T, 100)
    score_p = utils.score_fn(state)

    error = error_forward(ts[1:], forward_score, score_p, target, target)
    print("error:", error)
    print("time:", restored["time"])
    print("training", restored["training_config"])

    true_score = forward_score(0, target, 0.5, target).squeeze()
    trained_score = score_p(target, 0.5).squeeze()

    plt.scatter(true_score[:, 0], true_score[:, 1], c="b")
    plt.scatter(trained_score[:, 0], trained_score[:, 1], c="r")

    plt.savefig(f"./figures/score_{n_bases}_fourier_exact.pdf")

    key = jax.random.PRNGKey(23)
    reverse_sde = sdes.reverse(bm_sde, score_p)
    target = jnp.expand_dims(target, axis=0)
    backward_coeffs = sdes.simulate_traj(reverse_sde, target, 2, key)

    backward_traj = utils.inverse_fourier(backward_coeffs[0], 10)
    traj = backward_traj.reshape((-1, 10 * bm_sde.dim))

    plotting.plot_single_trajectory(traj, "Conditioned Brownian Motion")
    # target_plt = utils.inverse_fourier(target, 50)
    plt.scatter(target_pts[:, 0], target_pts[:, 1], c="b")
    plt.savefig(f"./figures/bm_{n_bases}_exact_fourier.pdf")
