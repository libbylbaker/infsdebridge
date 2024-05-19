import jax.numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils
from ml_collections import ConfigDict

from sdebridge import diffusion_bridge, sdes

if __name__ == "__main__":
    initial_butterfly = jnp.load("./data/tom_pts.npy")
    target_butterfly = jnp.load("./data/honrathi_pts.npy")

    n_bases = 24
    n_pts = initial_butterfly.shape[0]

    sde_config = ConfigDict(
        {
            "init_S": initial_butterfly,
            "Nt": n_bases,
            "n_grid": 64,
            "grid_range": [-1.5, 1.5],
            "alpha": 1.0,
            "sigma": 0.1,
            "T": 1.0,
            "n_bases": 50,
            "dim": 2,
        }
    )

    sde = FourierGaussianKernelSDE(sde_config)
    bridge = DiffusionBridge(sde)

    setup_params = {
        "network": {
            "output_dim": 2 * sde.dim * sde.n_bases,
            "time_embedding_dim": 96,
            "init_embedding_dim": 96,
            "act_fn": "silu",
            "encoder_layer_dims": [192, 96, 48, 24],
            "decoder_layer_dims": [24, 48, 96, 192],
            "batchnorm": True,
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 100,
            "num_batches_per_epoch": 300,
            "learning_rate": 1e-3,
            "warmup_steps": 2000,
        },
    }

    score_p_state = bridge.learn_p_score(
        initial_val=initial_diff_flatten,
        setup_params=setup_params,
    )

    score_p_ckpt = {
        "state": score_p_state,
        "training_config": setup_params,
    }

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(score_p_ckpt)
    orbax_checkpointer.save(
        "/home/gefan/Projects/sdebridge/sdebridge/ckpts/score_p_24_bases_retrain", score_p_ckpt, save_args=save_args
    )
    print("saved score_p_ckpt")
