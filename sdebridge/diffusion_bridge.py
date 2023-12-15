from functools import partial

import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm.notebook import tqdm

from . import utils
from .networks import ScoreNet


class DiffusionBridge:
    def __init__(
        self,
        drift: callable,
        diffusion: callable,
        dim: int,
        end_time: float,
        num_steps: int,
        rng: jax.random.PRNGKey = jax.random.PRNGKey(321),
        true_score_transition: callable = None,
        true_score_h: callable = None,
    ):
        self.f = drift
        self.g = diffusion
        self.d = dim
        self.T = end_time
        self.N = num_steps

        self.ts = jnp.linspace(0.0, self.T, self.N + 1)
        self.rng = rng
        self.true_score_transition = true_score_transition
        self.true_score_h = true_score_h

    def covariance(self, x, t):
        return jnp.dot(self.g(x, t), self.g(x, t).T)

    def inv_covariance(self, x, t):
        sigma = self.covariance(x, t)
        return jnp.linalg.inv(sigma)

    def divergence_covariance(self, x, t):
        jacobian_covariance = jax.jacfwd(self.covariance)
        return jnp.trace(jacobian_covariance(x, t))

    def euler_maruyama_scaled(
        self, initial_condition: jax.Array, drift: callable, diffusion: callable
    ) -> dict:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jax.Array): X(0)
            diffusion: function taking arguments  val, time
            drift: function taking arguments  val, time

        Returns:
            dict: {"trajectories": jax.Array, (B, N+1, d) forward non-bridge trajectories,
                   "scaled_brownians": jax.Array, (B, N, d) scaled stochastic updates for computing the gradients}
        """
        assert initial_condition.shape[-1] == self.d
        B = initial_condition.shape[0]  # batch size
        X = initial_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))  # (t0, t1, ..., tN)
        scaled_brownians = jnp.zeros(shape=(B, self.N, self.d))  # (t0, t1, ..., tN-1)
        trajectories = trajectories.at[:, 0, :].set(X)  # (X(t0))
        for t_idx in range(self.N):  # (0, 1, ... , N-1)
            t_now = self.ts[t_idx]  # (t0, t1, ..., tN-1)
            t_next = self.ts[t_idx + 1]  # (t1, t2, ..., tN)
            dt = t_next - t_now  # (t1-t0, t2-t1, ..., tN-tN-1)
            self.rng, _ = jax.random.split(self.rng)
            drift_step = dt * drift(val=X, time=t_now)
            brownian = jnp.sqrt(dt) * jax.random.normal(self.rng, shape=(B, self.d))
            diffusion_step = utils.batch_multi(diffusion(val=X, time=t_now), brownian)
            scaled_brownian = -utils.batch_multi(
                self.inv_covariance(X, t_now) / dt, diffusion_step
            )
            X = X + drift_step + diffusion_step  # Euler-Maruyama
            trajectories = trajectories.at[:, t_idx + 1, :].set(
                X
            )  # (X(t1), X(t2), ..., X(tN))
            scaled_brownians = scaled_brownians.at[:, t_idx, :].set(
                scaled_brownian
            )  # (dW(t0), dW(t1), ..., dW(tN-1))
        return {"trajectories": trajectories, "scaled_brownians": scaled_brownians}

    def euler_maruyama_scaled_bridge(
        self,
        initial_condition: jax.Array,
        terminal_condition: jax.Array,
        drift: callable,
        diffusion: callable,
    ) -> dict:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jax.Array): X(0)
            terminal_condition (jax.Array): X(T)
            diffusion: function taking arguments  val, time
            drift: function taking arguments  val, time

        Returns:
            dict: {"trajectories": jax.Array, (B, N+1, d) forward non-bridge trajectories,
                   "scaled_brownians": jax.Array, (B, N, d) scaled stochastic updates for computing the gradients}
        """

        non_bridge_process = self.euler_maruyama_scaled(
            initial_condition=initial_condition, diffusion=diffusion, drift=drift
        )
        trajectories = (
            non_bridge_process["trajectories"].at[:, -1, :].set(terminal_condition)
        )
        scaled_brownians = non_bridge_process["scaled_brownians"]
        t = self.ts[-2]
        dt = self.ts[-1] - self.ts[-2]
        last_val = trajectories[:, -2, :]
        last_drift = drift(val=last_val, time=t) * dt
        last_diffusion = terminal_condition - last_val - last_drift
        scaled_brownian = -utils.batch_multi(
            self.inv_covariance(last_val, t) / dt, last_diffusion
        )
        scaled_brownians = scaled_brownians.at[:, -1, :].set(scaled_brownian)
        return {"trajectories": trajectories, "scaled_brownians": scaled_brownians}

    def simulate_forward_process(self, initial_condition: jax.Array) -> dict:
        """Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jax.Array): X(0)

        Returns:
            dict: {"trajectories": jax.Array, (B, N+1, d) forward non-bridge trajectories,
                   "brownian_increments": jax.Array, (B, N, d) brownian increments for computing the gradients}
        """
        return self.euler_maruyama_scaled(
            initial_condition=initial_condition, drift=self.f, diffusion=self.g
        )

    def simulate_backward_bridge(
        self,
        initial_condition: jax.Array,
        terminal_condition: jax.Array,
        using_true_score: bool = False,
        score_p_state: utils.TrainState = None,
    ) -> dict:
        """Simulate the backward bridge process (Z*(t)):
            dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        Args:
            score_p_state (TrainState): s_{theta}(t, x)
            initial_condition (jax.Array): Z*(T) = X(0)
            terminal_condition (jax.Array): Z*(0) = X(T-t)
            using_true_score (bool): If True, use the predefined score transition function.

        Returns:
            dict: {"trajectories": jax.Array, (B, N+1, d) backward bridge trajectories,
                   "scaled_brownians": jax.Array, (B, N, d) brownian increments for computing the gradients}
        !!! N.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
        !!! N.B. scaled_brownians is also therefore exactly the opposite to what's given in simulate backward bridge!!!
        """

        terminal = terminal_condition.copy()

        def drift(val, time):
            inverted_time = self.T - time
            if using_true_score:
                score_p = self.true_score_transition(
                    val=val, start_val=terminal, time=inverted_time
                )
            else:
                score_p = utils.eval_score(state=score_p_state, x=val, t=inverted_time)
            rev_drift = -self.f(val, inverted_time)
            score_term = utils.batch_multi(self.covariance(val, inverted_time), score_p)
            div_term = self.divergence_covariance(val, inverted_time)
            return rev_drift + score_term + div_term

        def diffusion(val, time):
            reverse_time = self.T - time
            return self.g(val, reverse_time)

        # swapping initial and terminal conditions then mirroring at the end is equivalent to previous method
        process = self.euler_maruyama_scaled_bridge(
            initial_condition=terminal_condition,
            terminal_condition=initial_condition,
            drift=drift,
            diffusion=diffusion,
        )
        return {
            "trajectories": process["trajectories"][:, ::-1, :],
            "scaled_brownians": process["scaled_brownians"][:, ::-1, :],
        }

    def simulate_forward_bridge(
        self,
        initial_condition: jax.Array,
        terminal_condition: jax.Array,
        true_score: callable = None,
        score_p_state: utils.TrainState = None,
        score_p_star_state: utils.TrainState = None,
    ) -> dict:
        """Simulate the forward bridge process (X*(t)) which is the "backward of backward":
            dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

        Args:
            score_p_state (utils.TrainState): s_{theta}(t, x)
            score_p_star_state (utils.TrainState): s*_{theta}(t, x)
            initial_condition (jax.Array): X*(0)
            terminal_condition (jax.Array): X*(T)

        Returns:
            dict: {"trajectories": jax.Array, (B, N+1, d) forward bridge trajectories,
                   "scaled_brownians": None}
        """

        def drift(val, time):
            # todo: what should the behaviour be when t=0? I think it should return 1 for val=initial_cond and 0 otherwise.
            if time == 0:
                time = self.ts[1] * 0.1
            if true_score is not None:
                score_h = true_score(val=val, time=time)
            else:
                score_p = utils.eval_score(state=score_p_state, x=val, t=time)
                score_p_star = utils.eval_score(state=score_p_star_state, x=val, t=time)
                score_h = score_p_star - score_p
            orig_drift = self.f(val=val, time=time)
            bridge_term = utils.batch_multi(self.covariance(x=val, t=time), score_h)
            return orig_drift + bridge_term

        def diffusion(val, time):
            return self.g(val=val, time=time)

        processes = self.euler_maruyama_scaled_bridge(
            initial_condition=initial_condition,
            terminal_condition=terminal_condition,
            drift=drift,
            diffusion=diffusion,
        )
        return {"trajectories": processes["trajectories"], "scaled_brownians": None}

    def get_p_gradient(
        self,
        forward_trajectories: jax.Array,
        scaled_brownians: jax.Array,
        epsilon: float = 0.0,
    ) -> jax.Array:
        """Compute g(t_{m-1}, X_{m-1}, t_m, X_m) using the new expression for eq. (8):
            g(t_{m-1}, X_{m-1}, t_m, X_m) = - (\Sigma(t_{m-1}, X_{m-1}) * \delta t)^{-1} * \sigma(t_{m-1}, X_{m-1}) * \delta W(t_{m-1}, X_{m-1})

        Args:
            forward_trajectories (jax.Array): (B, N+1, d) forward non-bridge trajectories.
            scaled_brownians (jax.Array): (B, N, d) scaled stochastic updates for computing the gradients.
            epsilon (float, optional): a magical weight to enforce the initial constraint. Defaults to 0.0.

        Returns:
            jax.Array: (B, N, d) g(t_{m-1}, X_{m-1}, t_m, X_m)
        """
        assert forward_trajectories.shape[-1] == self.d
        B = forward_trajectories.shape[0]  # batch size
        X0 = forward_trajectories[:, 0, :]  # initial condition
        gradients = jnp.zeros(shape=(B, self.N, self.d))
        for t_idx in range(self.N):  # (0, 1, ..., N-2)
            # previous step for forward process
            X_m_minus_1 = forward_trajectories[:, t_idx, :]
            # current step for forward process
            X_m = forward_trajectories[:, t_idx + 1, :]
            if t_idx == self.N - 1:
                t_m = (
                    0.75 * self.ts[t_idx + 1] + 0.25 * self.ts[t_idx]
                )  # (t0, t1, ..., tN-1)
            else:
                t_m = self.ts[t_idx + 1]
            t_m_minus_1 = self.ts[t_idx]
            scaled_brownian = scaled_brownians[:, t_idx, :]
            additional_constraint = (
                epsilon
                * utils.batch_multi(
                    self.inv_covariance(X_m_minus_1, t_m_minus_1), (X0 - X_m)
                )
                / (self.T - t_m)
            )
            gradient = scaled_brownian - additional_constraint
            gradients = gradients.at[:, t_idx, :].set(gradient)
        return gradients

    def get_p_star_gradient(
        self,
        backward_trajectories: jax.Array,
        scaled_brownians: jax.Array,
        epsilon: float = 0.0,
    ) -> jax.Array:
        """Compute g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m) for eq. (18), but use the same trick as in get_p_gradient

        Args:
            backward_trajectories (jax.Array): (B, N+1, d) backward bridge trajectories
            scaled_brownians (jax.Array): (B, N, d) scaled stochastic updates for computing the gradients.
            epsilon (float, optional): a magical weight to enforce the initial constraint. Defaults to 1e-4.

        Returns:
            jax.Array: (B, N, d) g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m)
        """
        assert backward_trajectories.shape[-1] == self.d
        B = backward_trajectories.shape[0]  # batch size
        XT = backward_trajectories[:, -1, :]  # terminal condition
        gradients = jnp.zeros(shape=(B, self.N, self.d))
        for reverse_t_idx in range(self.N, 0, -1):  # (N, N-1, ..., 2)
            Z_m_minus_1 = backward_trajectories[
                :, reverse_t_idx, :
            ]  # previous step for backward process, i.e. next step for forward process
            Z_m = backward_trajectories[
                :, reverse_t_idx - 1, :
            ]  # current step for backward process, also for the forward process
            reverse_t_m_minus_1 = self.ts[reverse_t_idx]
            if reverse_t_idx == 1:
                reverse_t_m = 0.25 * (
                    self.ts[reverse_t_idx] - self.ts[reverse_t_idx - 1]
                )
            else:
                reverse_t_m = self.ts[reverse_t_idx - 1]
            dt = reverse_t_m_minus_1 - reverse_t_m
            scaled_brownian = scaled_brownians[:, reverse_t_idx - 1, :]
            additional_constraint = (
                epsilon
                * utils.batch_multi(
                    self.inv_covariance(Z_m_minus_1, reverse_t_m_minus_1), (XT - Z_m)
                )
                / reverse_t_m
            )
            gradient = scaled_brownian - additional_constraint
            gradients = gradients.at[:, reverse_t_idx - 1, :].set(gradient)
        return gradients

    def get_trajectories_generator(
        self,
        batch_size: int,
        process_type: str,
        initial_condition: jax.Array,
        terminal_condition: jax.Array,
        score_p_state: utils.TrainState,
        score_p_star_state: utils.TrainState,
    ) -> callable:
        assert process_type in ["forward", "backward_bridge", "forward_bridge"]
        assert initial_condition.shape[-1] == self.d

        def generator():
            initial_conditions = jnp.tile(initial_condition, reps=(batch_size, 1))
            terminal_conditions = (
                jnp.tile(terminal_condition, reps=(batch_size, 1))
                if terminal_condition is not None
                else None
            )
            while True:
                if process_type == "forward":
                    histories = self.simulate_forward_process(initial_conditions)
                elif process_type == "backward_bridge":
                    histories = self.simulate_backward_bridge(
                        initial_conditions,
                        terminal_conditions,
                        score_p_state=score_p_state,
                    )
                elif process_type == "forward_bridge":
                    histories = self.simulate_forward_bridge(
                        initial_conditions,
                        terminal_conditions,
                        score_p_state,
                        score_p_star_state,
                    )
                yield (histories["trajectories"], histories["scaled_brownians"])

        return generator

    def learn_p_score(
        self,
        initial_condition: jax.Array,
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
            initial_condition=initial_condition,
            terminal_condition=None,
            score_p_state=None,
            score_p_star_state=None,
        )
        dataset = utils.get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.float32, tf.float32),
            shape=[
                (training_params["batch_size"], self.N + 1, self.d),
                (training_params["batch_size"], self.N, self.d),
            ],
        )
        reduce_operation = (
            jax.vmap(jnp.mean, in_axes=-1)
            if reduce_mean
            else jax.vmap(0.5 * jnp.sum, in_axes=-1)
        )

        @jax.jit
        def train_step(state: utils.TrainState, batch: tuple) -> utils.TrainState:
            trajectories, scaled_brownians = batch
            ts = utils.flatten_batch(
                utils.unsqueeze(
                    jnp.tile(self.ts[1:], reps=(training_params["batch_size"], 1)),
                    axis=-1,
                )
            )  # (B*N, 1)
            score_p_gradients = self.get_p_gradient(
                trajectories, scaled_brownians
            )  # (B, N, d)
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
                loss = reduce_operation(loss)  # (B*N, d) -> (B*N, )
                if normalized:
                    normalization = jnp.mean(
                        score_p_gradients**2, axis=-1
                    )  # E[||\nabla\log p(xt|x0)||^2]
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
            score_p_net,
            self.rng,
            training_params["learning_rate"],
            [
                (training_params["batch_size"], self.d),
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
        initial_condition: jax.Array,
        terminal_condition: jax.Array,
        score_p_state: utils.TrainState,
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
            initial_condition=initial_condition,
            terminal_condition=terminal_condition,
            score_p_state=score_p_state,
            score_p_star_state=None,
        )
        dataset = utils.get_iterable_dataset(
            generator=data_generator,
            dtype=(tf.float32, tf.float32),
            shape=[
                (training_params["batch_size"], self.N + 1, self.d),
                (training_params["batch_size"], self.N, self.d),
            ],
        )
        reduce_operation = (
            jax.vmap(jnp.mean, in_axes=-1)
            if reduce_mean
            else jax.vmap(0.5 * jnp.sum, in_axes=-1)
        )

        @jax.jit
        def train_step(state: utils.TrainState, batch: tuple) -> utils.TrainState:
            trajectories, scaled_brownians = batch
            ts = utils.flatten_batch(
                utils.unsqueeze(
                    jnp.tile(self.ts[:-1], reps=(training_params["batch_size"], 1)),
                    axis=-1,
                )
            )  # (B*N, 1)
            score_p_star_gradients = self.get_p_star_gradient(
                trajectories, scaled_brownians
            )  # (B, N, d)
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
                if normalized:
                    normalization = jnp.mean(
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
                (training_params["batch_size"], self.d),
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
