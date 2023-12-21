from functools import partial

import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm

from . import utils
from .networks import ScoreNet
from .sde import SDE


class DiffusionBridge:
    def __init__(self, sde: SDE):
        self.sde = sde

    def simulate_forward_process(
        self, initial_val: jax.Array, rng: jax.Array = None
    ) -> dict:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jax.Array): X(0)

        Returns:
            result (dict): {"trajectories": jax.Array, (B, N, d) forward trajectories,
                            "scaled_stochastic_increments": jax.Array, (B, N, d) approximation of gradients,
                            "step_rngs": jax.Array, (B, N) random number generators}
        """
        results = utils.euler_maruyama(sde=self.sde, initial_val=initial_val, rng=rng)
        return results

    @jax.tree_util.Partial(jax.jit, static_argnums=(0,))
    def simulate_backward_bridge(
        self,
        initial_val: jax.Array,
        terminal_val: jax.Array,
        score_p: callable = None,
        rng: jax.Array = None,
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
        reverse_sde = self.sde.reverse_sde(score_p_density=score_p)
        results = utils.euler_maruyama(
            sde=reverse_sde,
            initial_val=terminal_val,
            terminal_val=initial_val,
            rng=rng,
        )
        return results

    @jax.tree_util.Partial(jax.jit, static_argnums=(0,))
    def simulate_forward_bridge(
        self,
        initial_val: jax.Array,
        terminal_val: jax.Array,
        score_h: callable = None,
        rng: jax.Array = None,
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

        bridge_sde = self.sde.bridge_sde(score_h_density=score_h)
        results = utils.euler_maruyama(
            sde=bridge_sde,
            initial_val=initial_val,
            terminal_val=terminal_val,
            rng=rng,
        )
        return results

    def get_trajectories_generator(
        self,
        batch_size: int,
        process_type: str,
        initial_val: jax.Array,
        terminal_val: jax.Array,
        score_p: callable = None,
        score_h: callable = None,
        rng: jax.Array = None,
    ) -> callable:
        assert process_type in ["forward", "backward_bridge", "forward_bridge"]
        assert initial_val.shape[-1] == self.sde.d

        def generator():
            local_rng = rng  # Assign rng to a local variable
            initial_vals = jnp.tile(initial_val, reps=(batch_size, 1))
            terminal_vals = (
                jnp.tile(terminal_val, reps=(batch_size, 1))
                if terminal_val is not None
                else None
            )
            while True:
                step_rng, local_rng = jax.random.split(local_rng)  # Update local_rng
                if process_type == "forward":
                    histories = self.simulate_forward_process(
                        initial_vals, rng=step_rng
                    )
                elif process_type == "backward_bridge":
                    histories = self.simulate_backward_bridge(
                        initial_vals, terminal_vals, score_p=score_p, rng=step_rng
                    )
                elif process_type == "forward_bridge":
                    histories = self.simulate_forward_bridge(
                        initial_vals, terminal_vals, score_h=score_h, rng=step_rng
                    )
                yield (
                    histories["trajectories"],
                    histories["scaled_stochastic_increments"],
                )

        return generator

    def learn_p_score(
        self,
        initial_val: jax.Array,
        reduce_mean: bool = True,
        setup_params: dict = None,
        loss_calibration: bool = False,
        weighted_norm: bool = True,
        rng: jax.Array = None,
        **kwargs,
    ) -> utils.TrainState:
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_net = ScoreNet(**net_params)
        data_generator = self.get_trajectories_generator(
            batch_size=training_params["batch_size"],
            process_type="forward",
            initial_val=initial_val,
            terminal_val=None,
            score_p=None,
            score_h=None,
            rng=rng,
        )

        iter_dataset = utils.get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.float32, tf.float32),
            shape=[
                (training_params["batch_size"], self.sde.N, self.sde.d),
                (training_params["batch_size"], self.sde.N, self.sde.d),
            ],
        )
        reduce_operation = (
            jax.vmap(jnp.mean, in_axes=-1)
            if reduce_mean
            else jax.vmap(0.5 * jnp.sum, in_axes=-1)
        )
        norm_operation = utils.weighted_norm if weighted_norm else utils.normal_norm

        @jax.jit
        def train_step(state: utils.TrainState, batch: tuple) -> utils.TrainState:
            trajectories, scaled_stochastic_increments = batch
            ts = utils.flatten_batch(
                utils.unsqueeze(
                    jnp.tile(self.sde.ts[1:], reps=(training_params["batch_size"], 1)),
                    axis=-1,
                )
            )  # (B*N, 1)
            score_p_gradients = scaled_stochastic_increments  # (B, N, d)
            score_p_gradients = utils.flatten_batch(score_p_gradients)  # (B*N, d)
            trajectories = utils.flatten_batch(trajectories)  # (B*N, d)
            covariances = jax.vmap(self.sde.covariance, in_axes=(0, None))(
                trajectories, ts
            )  # (B*N, d, d)
            if loss_calibration:
                score_p_density = partial(
                    self.sde.score_p_density,
                    init_val=jnp.tile(initial_val, reps=(trajectories.shape[0], 1)),
                    term_val=None,
                )
                true_p_scores = score_p_density(trajectories, ts)  # (B*N, d)

            def loss_fn(params) -> tuple:
                score_p_est, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, d)
                loss = norm_operation(
                    x=score_p_est - score_p_gradients, weight=covariances
                )
                if loss_calibration:
                    loss_cal1 = norm_operation(x=true_p_scores, weight=covariances)
                    loss_cal2 = norm_operation(x=score_p_gradients, weight=covariances)
                    loss = reduce_operation(loss + loss_cal1 - loss_cal2)
                else:
                    loss = reduce_operation(loss)  # (B*N, d) -> (B*N, )
                loss = 0.5 * self.sde.dt * jnp.mean(loss, axis=0)
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state

        network_rng, _ = jax.random.split(rng)
        state = utils.create_train_state(
            score_p_net,
            network_rng,
            training_params["learning_rate"],
            [
                (training_params["batch_size"], self.sde.d),
                (training_params["batch_size"], 1),
            ],
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
        initial_val: jax.Array,
        terminal_val: jax.Array,
        score_p: callable,
        reduce_mean: bool = True,
        setup_params: dict = None,
        weighted_norm: bool = True,
        loss_calibration: bool = False,
        rng: jax.Array = None,
    ) -> utils.TrainState:
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_star_net = ScoreNet(**net_params)

        data_generator = self.get_trajectories_generator(
            batch_size=training_params["batch_size"],
            process_type="backward_bridge",
            initial_val=initial_val,
            terminal_val=terminal_val,
            score_p=score_p,
            score_h=None,
            rng=rng,
        )
        iter_dataset = utils.get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.float32, tf.float32),
            shape=[
                (training_params["batch_size"], self.sde.N, self.sde.d),
                (training_params["batch_size"], self.sde.N, self.sde.d),
            ],
        )
        reduce_operation = (
            jax.vmap(jnp.mean, in_axes=-1)
            if reduce_mean
            else jax.vmap(0.5 * jnp.sum, in_axes=-1)
        )

        norm_operation = utils.weighted_norm if weighted_norm else utils.normal_norm

        @jax.jit
        def train_step(state: utils.TrainState, batch: tuple) -> utils.TrainState:
            trajectories, scaled_stochastic_increments = batch
            ts = utils.flatten_batch(
                utils.unsqueeze(
                    jnp.tile(
                        self.sde.T - self.sde.ts[:-1],
                        reps=(training_params["batch_size"], 1),
                    ),  # !!! the backward trajectories are in the reverse order, so we need inverted time series.
                    axis=-1,
                )
            )  # (B*N, 1)
            score_p_star_gradients = scaled_stochastic_increments
            score_p_star_gradients = utils.flatten_batch(
                score_p_star_gradients
            )  # (B*N, d)
            trajectories = utils.flatten_batch(trajectories)  # (B*N, d)
            covariances = jax.vmap(self.sde.covariance, in_axes=(0, None))(
                trajectories, ts
            )  # (B*N, d, d)
            if loss_calibration:
                score_h_density = partial(
                    self.sde.score_h_density,
                    init_val=None,
                    term_val=jnp.tile(terminal_val, reps=(trajectories.shape[0], 1)),
                )
                score_p_density = partial(
                    self.sde.score_p_density,
                    init_val=jnp.tile(initial_val, reps=(trajectories.shape[0], 1)),
                    term_val=None,
                )
                true_p_star_scores = score_p_density(
                    trajectories, ts
                ) + score_h_density(
                    trajectories, ts
                )  # (B*N, d)

            def loss_fn(params) -> tuple:
                score_p_star_est, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, d)
                loss = norm_operation(
                    x=score_p_star_est - score_p_star_gradients, weight=covariances
                )
                if loss_calibration:
                    loss_cal1 = norm_operation(x=true_p_star_scores, weight=covariances)
                    loss_cal2 = norm_operation(
                        x=score_p_star_gradients, weight=covariances
                    )
                    loss = reduce_operation(loss + loss_cal1 - loss_cal2)
                else:
                    loss = reduce_operation(loss)  # (B*N, d) -> (B*N, )
                loss = 0.5 * self.sde.dt * jnp.mean(loss, axis=0)
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state

        network_rng, _ = jax.random.split(rng)
        state = utils.create_train_state(
            score_p_star_net,
            network_rng,
            training_params["learning_rate"],
            [
                (training_params["batch_size"], self.sde.d),
                (training_params["batch_size"], 1),
            ],
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
