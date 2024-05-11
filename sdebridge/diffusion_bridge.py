from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import tensorflow as tf
from einops import rearrange, repeat
from tqdm import tqdm

from sdebridge.networks.score_unet import ScoreUNet
from sdebridge.sde import SDE, bridge, reverse
from sdebridge.utils import (
    complex_weighted_norm_square,
    create_train_state,
    get_iterable_dataset,
    weighted_norm_square,
)


class DiffusionBridge:
    def __init__(self, sde: SDE):
        self.sde = sde

    @partial(jax.jit, static_argnums=(0, 2))
    def simulate_forward_process(
        self,
        initial_val: jnp.ndarray,
        num_batches: int,
        rng_key: jax.Array,
    ) -> jax.Array:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_val (jnp.ndarray): X(0)
            rng_key (jax.Array): random number generator
            num_batches (int): number of batches to simulate

        Returns:
            result: trajectories: jax.Array, (B, N, d) forward trajectories
        """
        return self.sde.simulate_trajectories(initial_val, num_batches, rng_key)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def simulate_backward_bridge(
        self,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_p: Callable,
        num_batches: int,
        rng_key: jax.Array,
    ) -> jax.Array:
        """Simulate the backward bridge process (Z*(t)):
            dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        Args:
            initial_val (jax.Array): X(0) = Z*(T)
            terminal_val (jax.Array): X(T) = Z*(0)
            score_p (callable): nabla log p(x, t), either a closed form or a neural network.

        Returns:
            results: trajectories: jax.Array, (B, N, d) backward bridge trajectories
        !!! N.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
        """

        reverse_sde = reverse(self.sde, score_fun=score_p)
        return reverse_sde.simulate_trajectories(initial_val, num_batches, rng_key)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def simulate_forward_bridge(
        self,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_h: Callable,
        num_batches: int,
        rng_key: jax.Array,
    ) -> jax.Array:
        """Simulate the forward bridge process (X*(t)) which is the "backward of backward":
            dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

        Args:
            initial_val (jax.Array): X*(0)
            terminal_val (jax.Array): X*(T)
            score_h (callable): nabla log h(x, t), either a closed form or a neural network.

        Returns:
            results: "trajectories": jax.Array, (B, N, d) forward bridge trajectories (in normal order)
        """

        bridge_sde = bridge(self.sde, score_fun=score_h)
        return bridge_sde.simulate_trajectories(initial_val, num_batches, rng_key)

    def forward_generator(
        self,
        key: jax.Array,
        batch_size: int,
        initial_val: jnp.ndarray,
    ) -> callable:
        def generator():
            subkey = key
            while True:
                step_key, subkey = jax.random.split(subkey)
                trajs = self.sde.simulate_trajectories(
                    initial_val, key=step_key, num_batches=batch_size
                )
                grads, covs = self.sde.grad_and_covariance(trajs)
                yield trajs[:, 1:], grads, covs,

        return generator

    def backward_bridge_generator(
        self,
        rng_key: jax.Array,
        batch_size: int,
        initial_val: jnp.ndarray,
        score_p: Callable,
    ) -> callable:
        reverse_sde = reverse(self.sde, score_p)

        def generator():
            subkey = rng_key
            while True:
                step_key, subkey = jax.random.split(subkey)
                trajs = reverse_sde.simulate_trajectories(initial_val, batch_size, step_key)
                grads, covs = reverse_sde.grad_and_covariance(trajs)
                yield trajs[:, 1:], grads, covs

        return generator

    def learn_p_score(
        self,
        initial_val: jnp.ndarray,
        rng_key: jax.Array,
        setup_params: dict = None,
    ):
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        b_size = training_params["batch_size"]
        load_size = training_params["load_size"]
        net = setup_params["neural_net"]

        score_p_net = net(**net_params)

        data_rng_key, network_init_rng_key = jax.random.split(rng_key)
        data_generator = self.forward_generator(data_rng_key, load_size, initial_val)
        iter_dataset = get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.float64, tf.float64, tf.float64),
            shape=[
                (load_size, self.sde.N - 1, self.sde.n_bases * self.sde.dim),
                (load_size, self.sde.N - 1, self.sde.n_bases * self.sde.dim),
                (
                    load_size,
                    self.sde.N - 1,
                    self.sde.n_bases * self.sde.dim,
                    self.sde.n_bases * self.sde.dim,
                ),
            ],
        )

        @jax.jit
        def train_step(state, batch: tuple):
            trajectories, gradients, covariances = batch  # (B, N, 2*n_bases)
            ts = rearrange(
                repeat(self.sde.ts[1:], "n -> b n", b=b_size),
                "b n -> (b n) 1",
                b=b_size,
            )

            trajectories = rearrange(trajectories, "b n d -> (b n) d")  # (B*N, 2*n_bases)
            gradients = rearrange(gradients, "b n d -> (b n) d")  # (B*N, 2*n_bases)
            covariances = rearrange(
                covariances, "b n d1 d2 -> (b n) d1 d2"
            )  # (B*N, 2*n_bases, 2*n_bases)

            def loss_fn_complex(params) -> tuple:
                score, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, 2*n_bases)
                score_real, score_imag = jnp.split(score, 2, axis=-1)
                score_complex = score_real + 1j * score_imag
                loss = complex_weighted_norm_square(
                    x=score_complex - gradients, weight=covariances
                )
                loss = 0.5 * self.sde.dt * jnp.mean(loss, axis=0)
                return loss, updates

            def loss_fn(params) -> tuple:
                score, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, 2*n_bases)
                loss = weighted_norm_square(
                    x=score - gradients, weight=covariances
                )  # (B*N, d) -> (B*N, )
                loss = 0.5 * self.sde.dt * jnp.mean(loss, axis=0)
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])
            step_rng_key, _ = jax.random.split(state.key)
            state = state.replace(key=step_rng_key)

            return state, loss

        state = create_train_state(
            model=score_p_net,
            rng_key=network_init_rng_key,
            input_shapes=[
                (b_size, self.sde.dim * self.sde.n_bases),
                (b_size, 1),
            ],
            learning_rate=training_params["learning_rate"],
            warmup_steps=training_params["warmup_steps"],
            decay_steps=training_params["num_epochs"] * training_params["num_batches_per_epoch"],
        )
        pbar = tqdm(
            range(training_params["num_epochs"]),
            desc="Training",
            leave=True,
            unit="epoch",
            total=training_params["num_epochs"],
        )
        for i in pbar:
            total_loss = 0
            load = next(iter_dataset)
            for b in range(int(load_size / b_size)):
                tmp1, tmp2, tmp3 = load
                batch = (
                    tmp1[b * b_size : (b + 1) * b_size],
                    tmp2[b * b_size : (b + 1) * b_size],
                    tmp3[b * b_size : (b + 1) * b_size],
                )
                state, loss = train_step(state, batch)
                total_loss += loss
            epoch_loss = total_loss / training_params["num_batches_per_epoch"]
            pbar.set_postfix(Epoch=i + 1, loss=f"{epoch_loss:.4f}")

        return state

    def learn_p_star_score(
        self,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_p: Callable,
        rng_key: jax.Array,
        setup_params: dict = None,
    ):
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_star_net = ScoreUNet(**net_params)

        b_size = training_params["batch_size"]
        load_size = training_params["load_size"]

        data_rng_key, network_init_rng_key = jax.random.split(rng_key)

        data_generator = self.backward_bridge_generator(
            rng_key=data_rng_key,
            batch_size=load_size,
            initial_val=initial_val,
            terminal_val=terminal_val,
            score_p=score_p,
        )

        iter_dataset = get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.complex64, tf.complex64, tf.complex64),
            shape=[
                (b_size, self.sde.N, self.sde.n_bases * self.sde.dim),
                (b_size, self.sde.N, self.sde.n_bases * self.sde.dim),
                (
                    b_size,
                    self.sde.N,
                    self.sde.n_bases * self.sde.dim,
                    self.sde.n_bases * self.sde.dim,
                ),
            ],
        )

        @jax.jit
        def train_step(state, batch: tuple):
            trajectories, gradients, covariances = batch  # (B, N, 2*n_bases)
            ts = rearrange(
                repeat(
                    self.sde.T - self.sde.ts[:-1], "n -> b n", b=b_size
                ),  # !!! the backward trajectories are in the reverse order, so we need inverted time series.
                "b n -> (b n) 1",
                b=b_size,
            )

            trajectories = rearrange(trajectories, "b n d -> (b n) d")  # (B*N, 2*n_bases)
            gradients = rearrange(gradients, "b n d -> (b n) d")  # (B*N, 2*n_bases)
            covariances = rearrange(
                covariances, "b n d1 d2 -> (b n) d1 d2"
            )  # (B*N, 2*n_bases, 2*n_bases)

            def loss_fn(params) -> tuple:
                score, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x_complex=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, 2*n_bases)
                score_real, score_imag = jnp.split(score, 2, axis=-1)
                score_complex = score_real + 1j * score_imag
                loss = complex_weighted_norm_square(
                    x=score_complex - gradients, weight=covariances
                )
                loss = 0.5 * self.sde.dt * jnp.mean(loss, axis=0)
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])
            step_rng_key, _ = jax.random.split(state.key)
            state = state.replace(key=step_rng_key)

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state

        state = create_train_state(
            model=score_p_star_net,
            rng_key=network_init_rng_key,
            input_shapes=[
                (b_size, self.sde.dim * self.sde.n_bases),
                (b_size, 1),
            ],
            learning_rate=training_params["learning_rate"],
            warmup_steps=training_params["warmup_steps"],
            decay_steps=training_params["num_epochs"] * training_params["num_batches_per_epoch"],
        )
        pbar = tqdm(
            range(training_params["num_epochs"]),
            desc="Training",
            leave=True,
            unit="epoch",
            total=training_params["num_epochs"],
        )
        for i in pbar:
            load = next(iter_dataset)
            for b in range(training_params["num_batches_per_epoch"]):
                batch = load[b * b_size : (b + 1) * b_size]
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i + 1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state
