from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
from einops import rearrange, repeat

from networks import ScoreUNet
from sde import SDE
from solver import euler_maruyama
from utils import get_iterable_dataset, create_train_state, complex_weighted_norm_square

GDRK = jax.random.PRNGKey(0)

class DiffusionBridge:
    def __init__(self, sde: SDE):
        self.sde = sde

    @partial(jax.jit, static_argnums=(0, 2))
    def simulate_forward_process(
        self,
        initial_val: jnp.ndarray,
        num_batches: int,
        rng_key: jax.Array = GDRK,
    ) -> dict:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jnp.ndarray): X(0)
            rng (jax.Array): random number generator
            num_batches (int): number of batches to simulate

        Returns:
            result (dict): {"trajectories": jax.Array, (B, N, d) forward trajectories,
                            "scaled_stochastic_increments": jax.Array, (B, N, d) approximation of gradients,
                            "step_rngs": jax.Array, (B, N) random number generators}
        """
        initial_vals = jnp.tile(initial_val, reps=(num_batches, 1))
        results = euler_maruyama(
            sde=self.sde, initial_vals=initial_vals, terminal_vals=None, rng_key=rng_key
        )
        return results

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def simulate_backward_bridge(
        self,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_p: Callable,
        num_batches: int,
        rng_key: jax.Array = GDRK,
    ) -> dict:
        """Simulate the backward bridge process (Z*(t)):
            dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        Args:
            initial_val (jax.Array): X(0) = Z*(T)
            terminal_val (jax.Array): X(T) = Z*(0)
            score_p (callable): \nabla\log p(x, t), either a closed form or a neural network.

        Returns:
            results: {"trajectories": jax.Array, (B, N, d) backward bridge trajectories,
                      "scaled_stochastic_increments": jax.Array, (B, N, d) approximation of gradients,
                      "step_rngs": jax.Array, (B, N) random number generators}
        !!! N.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
        !!! N.B. scaled_stochastic_increments is also therefore exactly the opposite to what's given in simulate backward bridge!!!
        """
        initial_vals = jnp.tile(initial_val, reps=(num_batches, 1))
        terminal_vals = jnp.tile(terminal_val, reps=(num_batches, 1))

        reverse_sde = self.sde.reverse_sde(score_func=score_p)
        results = euler_maruyama(
            sde=reverse_sde,
            initial_vals=terminal_vals,  # NOTE: since here the reverse bridge is simulated, we need to swap the initial and terminal values.
            terminal_vals=initial_vals,
            rng_key=rng_key,
        )
        return results

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def simulate_forward_bridge(
        self,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_h: Callable,
        num_batches: int,
        rng_key: jax.Array = GDRK,
    ) -> dict:
        """Simulate the forward bridge process (X*(t)) which is the "backward of backward":
            dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

        Args:
            initial_val (jax.Array): X*(0)
            terminal_val (jax.Array): X*(T)
            score_h (callable): \nabla\log h(x, t), either a closed form or a neural network.

        Returns:
            results: {"trajectories": jax.Array, (B, N, d) forward bridge trajectories (in normal order)
                      "scaled_stochastic_increments": jax.Array, (B, N, d) approximation of gradients (not used anymore),
                      "step_rngs": jax.Array, (B, N) random number generators}
        """

        initial_vals = jnp.tile(initial_val, reps=(num_batches, 1))
        terminal_vals = jnp.tile(terminal_val, reps=(num_batches, 1))

        bridge_sde = self.sde.bridge_sde(score_func=score_h)
        results = euler_maruyama(
            sde=bridge_sde,
            initial_vals=initial_vals,
            terminal_vals=terminal_vals,
            rng_key=rng_key,
        )
        return results

    def get_trajectories_generator(
        self,
        batch_size: int,
        process_type: str,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_p: Callable,
        score_h: Callable,
        rng_key: jax.Array = GDRK,
    ) -> callable:
        assert process_type in ["forward", "backward_bridge", "forward_bridge"]

        def generator():
            nonlocal rng_key
            while True:
                step_rng_key, rng_key = jax.random.split(rng_key)
                if process_type == "forward":
                    histories = self.simulate_forward_process(
                        initial_val, rng_key=step_rng_key, num_batches=batch_size
                    )
                elif process_type == "backward_bridge":
                    histories = self.simulate_backward_bridge(
                        initial_val,
                        terminal_val,
                        score_p=score_p,
                        rng_key=step_rng_key,
                        num_batches=batch_size,
                    )
                elif process_type == "forward_bridge":
                    histories = self.simulate_forward_bridge(
                        initial_val,
                        terminal_val,
                        score_h=score_h,
                        rng_key=step_rng_key,
                        num_batches=batch_size,
                    )
                rng_key = histories["last_key"]
                yield (
                    histories["trajectories"],
                    histories["gradients"],
                    histories["covariances"]
                )

        return generator

    def learn_p_score(
        self,
        initial_val: jnp.ndarray,
        setup_params: dict = None,
        rng_key: jax.Array = GDRK,
    ):
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_net = ScoreUNet(**net_params)

        b_size = training_params["batch_size"]

        data_rng_key, network_init_rng_key = jax.random.split(rng_key)

        data_generator = self.get_trajectories_generator(
            batch_size=b_size,
            process_type="forward",
            initial_val=initial_val,
            terminal_val=None,
            score_p=None,
            score_h=None,
            rng_key=data_rng_key,
        )

        iter_dataset = get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.complex64, tf.complex64, tf.complex64),
            shape=[
                (b_size, self.sde.N, self.sde.n_bases*self.sde.dim),
                (b_size, self.sde.N, self.sde.n_bases*self.sde.dim),
                (b_size, self.sde.N, self.sde.n_bases*self.sde.dim, self.sde.n_bases*self.sde.dim)
            ],
        )

        @jax.jit
        def train_step(state, batch: tuple):
            trajectories, gradients, covariances = batch # (B, N, 2*n_bases)
            ts = rearrange(
                repeat(self.sde.ts[1:], 'n -> b n', b=b_size), 
                'b n -> (b n) 1', 
                b=b_size
            )

            trajectories = rearrange(trajectories, 'b n d -> (b n) d')  # (B*N, 2*n_bases)
            gradients = rearrange(gradients, 'b n d -> (b n) d')  # (B*N, 2*n_bases)
            covariances = rearrange(covariances, 'b n d1 d2 -> (b n) d1 d2')  # (B*N, 2*n_bases, 2*n_bases)
      
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
            model=score_p_net,
            rng_key=network_init_rng_key,
            input_shapes=[
                (b_size, self.sde.dim*self.sde.n_bases),
                (b_size, 1),
            ],
            learning_rate=training_params["learning_rate"],
            warmup_steps=training_params["warmup_steps"],
            decay_steps=training_params["num_epochs"]
            * training_params["num_batches_per_epoch"],
        )
        pbar = tqdm(
            range(training_params["num_epochs"]),
            desc="Training",
            leave=True,
            unit="epoch",
            total=training_params["num_epochs"],
        )
        for i in pbar:
            for _ in range(training_params["num_batches_per_epoch"]):
                batch = next(iter_dataset)
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i + 1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state

    def learn_p_star_score(
        self,
        initial_val: jnp.ndarray,
        terminal_val: jnp.ndarray,
        score_p: Callable,
        setup_params: dict = None,
        rng_key: jax.Array = GDRK,
    ):
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_star_net = ScoreUNet(**net_params)

        b_size = training_params["batch_size"]

        data_rng_key, network_init_rng_key = jax.random.split(rng_key)

        data_generator = self.get_trajectories_generator(
            batch_size=b_size,
            process_type="backward_bridge",
            initial_val=initial_val,
            terminal_val=terminal_val,
            score_p=score_p,
            score_h=None,
            rng_key=data_rng_key,
        )

        iter_dataset = get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.complex64, tf.complex64, tf.complex64),
            shape=[
                (b_size, self.sde.N, self.sde.n_bases*self.sde.dim),
                (b_size, self.sde.N, self.sde.n_bases*self.sde.dim),
                (b_size, self.sde.N, self.sde.n_bases*self.sde.dim, self.sde.n_bases*self.sde.dim)
            ],
        )

        @jax.jit
        def train_step(state, batch: tuple):
            trajectories, gradients, covariances = batch # (B, N, 2*n_bases)
            ts = rearrange(
                repeat(self.sde.T - self.sde.ts[:-1], 'n -> b n', b=b_size),  # !!! the backward trajectories are in the reverse order, so we need inverted time series.
                'b n -> (b n) 1', 
                b=b_size
            )

            trajectories = rearrange(trajectories, 'b n d -> (b n) d')  # (B*N, 2*n_bases)
            gradients = rearrange(gradients, 'b n d -> (b n) d')  # (B*N, 2*n_bases)
            covariances = rearrange(covariances, 'b n d1 d2 -> (b n) d1 d2')  # (B*N, 2*n_bases, 2*n_bases)

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
                (b_size, self.sde.dim*self.sde.n_bases),
                (b_size, 1),
            ],
            learning_rate=training_params["learning_rate"],
            warmup_steps=training_params["warmup_steps"],
            decay_steps=training_params["num_epochs"]
            * training_params["num_batches_per_epoch"],
        )
        pbar = tqdm(
            range(training_params["num_epochs"]),
            desc="Training",
            leave=True,
            unit="epoch",
            total=training_params["num_epochs"],
        )
        for i in pbar:
            for _ in range(training_params["num_batches_per_epoch"]):
                batch = next(iter_dataset)
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i + 1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state
