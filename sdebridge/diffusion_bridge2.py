from functools import partial

import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm.notebook import tqdm

from . import utils
from .networks import ScoreNet
from .sde import SDE


class DiffusionBridge:
    def __init__(self, sde: SDE, rng: jax.Array):
        self.sde = sde
        self.rng = rng

    def simulate_forward_process(self, initial_val: jax.Array) -> dict:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jax.Array): X(0)

        Returns:
            result (dict): {"trajectories": jax.Array, (B, N, d) forward trajectories,
                            "scaled_stochastics": jax.Array, (B, N, d) approximation of gradients}
        """
        results = utils.euler_maruyama(
            sde=self.sde, initial_val=initial_val, rng=self.rng
        )
        self.rng = results["rng"][-1]
        return results

    def simulate_backward_bridge(
        self, initial_val: jax.Array, terminal_val: jax.Array, score_p: callable = None
    ) -> dict:
        """Simulate the backward bridge process (Z*(t)):
            dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        Args:
            initial_val (jax.Array): X(0) = Z*(T)
            terminal_val (jax.Array): X(T) = Z*(0)
            score_p (callable): \nabla\log p(x, t), either a closed form or a neural network.

        Returns:
            results: {"trajectories": jax.Array, (B, N, d) backward bridge trajectories,
                      "scaled_stochastics": jax.Array, (B, N, d) approximation of gradients}
        !!! N.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
        !!! N.B. scaled_stochastics is also therefore exactly the opposite to what's given in simulate backward bridge!!!
        """
        reverse_sde = self.sde.reverse_sde(score_p_density=score_p)

        results = utils.euler_maruyama(
            sde=reverse_sde,
            initial_val=terminal_val,
            rng=self.rng,
            terminal_val=initial_val,
        )
        self.rng = results["rng"][-1]
        return results

    def simulate_forward_bridge(
        self, initial_val: jax.Array, terminal_val: jax.Array, score_h: callable = None
    ) -> dict:
        """Simulate the forward bridge process (X*(t)) which is the "backward of backward":
            dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

        Args:
            initial_val (jax.Array): X*(0)
            terminal_val (jax.Array): X*(T)
            score_h (callable): \nabla\log h(x, t), either a closed form or a neural network.

        Returns:
            results: {"trajectories": jax.Array, (B, N, d) forward bridge trajectories (in normal order)
                      "scaled_stochastics": jax.Array, (B, N, d) approximation of gradients (not used anymore)}
        """

        bridge_sde = self.sde.bridge_sde(score_h_density=score_h)
        results = utils.euler_maruyama(
            sde=bridge_sde,
            initial_val=initial_val,
            terminal_val=terminal_val,
            rng=self.rng,
        )
        self.rng = results["rng"][-1]
        return results

    def get_trajectories_generator(
        self,
        batch_size: int,
        process_type: str,
        initial_val: jax.Array,
        terminal_val: jax.Array,
        score_p: callable = None,
        score_h: callable = None,
    ) -> callable:
        assert process_type in ["forward", "backward_bridge", "forward_bridge"]
        assert initial_val.shape[-1] == self.sde.d

        def generator():
            initial_vals = jnp.tile(initial_val, reps=(batch_size, 1))
            terminal_vals = (
                jnp.tile(terminal_val, reps=(batch_size, 1))
                if terminal_val is not None
                else None
            )
            while True:
                if process_type == "forward":
                    histories = self.simulate_forward_process(initial_vals)
                elif process_type == "backward_bridge":
                    histories = self.simulate_backward_bridge(
                        initial_vals,
                        terminal_vals,
                        score_p=score_p,
                    )
                elif process_type == "forward_bridge":
                    histories = self.simulate_forward_bridge(
                        initial_vals,
                        terminal_vals,
                        score_h=score_h,
                    )
                yield (histories["trajectories"], histories["scaled_stochastics"])

        return generator

    def learn_p_score(
        self,
        initial_val: jax.Array,
        normalized: bool = False,
        reduce_mean: bool = True,
        setup_params: dict = None,
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
        )
        dataset = utils.get_iterable_dataset(
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
        normalize = partial(self.sde.score_p_density, init_val=initial_val)

        @jax.jit
        def train_step(state: utils.TrainState, batch: tuple) -> utils.TrainState:
            trajectories, scaled_stochastics = batch
            ts = utils.flatten_batch(
                utils.unsqueeze(
                    jnp.tile(self.sde.ts[1:], reps=(training_params["batch_size"], 1)),
                    axis=-1,
                )
            )  # (B*N, 1)
            score_p_gradients = scaled_stochastics[:, 1:, :]  # (B, N, d)
            score_p_gradients = utils.flatten_batch(score_p_gradients)  # (B*N, d)
            trajectories = utils.flatten_batch(trajectories[:, 1:, :])  # (B*N, d)

            def loss_fn(params) -> tuple:
                score_p_est, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, d)
                loss = jnp.square((score_p_est - score_p_gradients))
                if normalized:  # !!! The normalization here is not correct now.
                    normalization = jax.vmap(normalize)(trajectories, ts)
                    loss = loss * normalization
                loss = reduce_operation(loss)  # (B*N, d) -> (B*N, )
                loss = jnp.mean(loss, axis=0)
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state

        state = utils.create_train_state(
            score_p_net,
            self.rng,
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
                batch = next(dataset)
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i + 1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state

    def learn_p_star_score(
        self,
        initial_val: jax.Array,
        terminal_val: jax.Array,
        score_p: callable,
        normalized: bool = False,
        reduce_mean: bool = True,
        setup_params: dict = None,
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
        )
        dataset = utils.get_iterable_dataset(
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

        @jax.jit
        def train_step(state: utils.TrainState, batch: tuple) -> utils.TrainState:
            trajectories, scaled_stochastics = batch
            ts = utils.flatten_batch(
                utils.unsqueeze(
                    jnp.tile(
                        self.sde.T - self.sde.ts[:-1],
                        reps=(training_params["batch_size"], 1),
                    ),  # !!! the backward trajectories are in the reverse order, so we need inverted time series.
                    axis=-1,
                )
            )  # (B*N, 1)
            score_p_star_gradients = scaled_stochastics[:, :-1, :]
            score_p_star_gradients = utils.flatten_batch(
                score_p_star_gradients
            )  # (B*N, d)
            trajectories = utils.flatten_batch(trajectories[:, :-1, :])  # (B*N, d)

            def loss_fn(params) -> tuple:
                score_p_star_est, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x=trajectories,
                    t=ts,
                    train=True,
                    mutable=["batch_stats"],
                )  # (B*N, d)
                loss = jnp.square((score_p_star_est - score_p_star_gradients))
                loss = reduce_operation(loss)  # (B*N, d) -> (B*N, )
                if normalized:  # !!! Same wrong normalization.
                    normalization = jax.vmap(self.sde.score_p_density, in_axes=(0, 0))(
                        trajectories, ts
                    )
                    jnp.mean(
                        score_p_star_gradients**2, axis=-1
                    )  # E[||\nabla\log p*(xt|x0)||^2]
                    loss = loss / normalization
                loss = jnp.mean(loss, axis=0)
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state

        state = utils.create_train_state(
            score_p_star_net,
            self.rng,
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
                batch = next(dataset)
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i + 1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state
