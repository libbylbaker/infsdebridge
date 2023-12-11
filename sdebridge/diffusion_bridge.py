from tqdm.notebook import tqdm
from functools import partial

import jax
import jax.numpy as jnp

from .utils import *
from .networks import ScoreNet

class DiffusionBridge:
    def __init__(self,
                 drift: callable,
                 diffusion: callable,
                 dim: int,
                 end_time: float,
                 num_steps: int,
                 rng: jax.random.PRNGKey=jax.random.PRNGKey(321)):
        self.f = drift
        self.g = diffusion
        self.Sigma = lambda x, t: jnp.dot(self.g(x, t), self.g(x, t).T)
        self.inv_Sigma = lambda x, t: jnp.linalg.inv(self.Sigma(x, t))
        self.div_Sigma = lambda x, t: jnp.trace(jax.jacfwd(self.Sigma)(x, t))
        self.d = dim
        self.T = end_time
        self.N = num_steps
        self.ts = jnp.linspace(0.0, self.T, self.N+1)
        self.rng = rng

    def simulate_forward_process(self, 
                                 initial_condition: jnp.ndarray) -> dict:
        """ Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jnp.ndarray): X(0) 

        Returns:
            dict: {"trajectories": jnp.ndarray, (B, N+1, d) forward non-bridge trajectories,
                   "scaled_brownians": jnp.ndarray, (B, N, d) scaled stochastic updates for computing the gradients}
        """        
        assert initial_condition.shape[-1] == self.d
        B = initial_condition.shape[0]                                  # batch size
        X = initial_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))         # (t0, t1, ..., tN)
        scaled_brownians = jnp.zeros(shape=(B, self.N, self.d))         # (t0, t1, ..., tN-1)
        trajectories = trajectories.at[:, 0, :].set(X)                  # (X(t0))                
        for t_idx in range(self.N):                                     # (0, 1, ... , N-1)
            t_now = self.ts[t_idx]                                      # (t0, t1, ..., tN-1)
            t_next = self.ts[t_idx+1]                                   # (t1, t2, ..., tN)
            dt = t_next - t_now                                         # (t1-t0, t2-t1, ..., tN-tN-1)
            self.rng, _ = jax.random.split(self.rng)
            drift = dt * self.f(X, t_now)
            brownian = jnp.sqrt(dt) * jax.random.normal(self.rng, shape=(B, self.d))
            diffusion = sb_multi(self.g(X, t_now), brownian)
            scaled_brownian = -sb_multi(self.inv_Sigma(X, t_now) / dt, diffusion) 
            X = X + drift + diffusion                                   # Euler-Maruyama
            trajectories = trajectories.at[:, t_idx + 1, :].set(X)      # (X(t1), X(t2), ..., X(tN))
            scaled_brownians = scaled_brownians.at[:, t_idx, :].set(scaled_brownian)   # (dW(t0), dW(t1), ..., dW(tN-1))
        return {"trajectories": trajectories, "scaled_brownians": scaled_brownians}
    
    def simulate_backward_bridge(self, 
                                 score_p_state: TrainState,
                                 initial_condition: jnp.ndarray, 
                                 terminal_condition: jnp.ndarray) -> dict:
        """ Simulate the backward bridge process (Z*(t)):
            dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        Args:
            score_p_state (TrainState): s_{theta}(t, x)
            initial_condition (jnp.ndarray): Z*(0) = X(T-t)
            terminal_condition (jnp.ndarray): Z*(T) = X(0)

        Returns:
            dict: {"trajectories": jnp.ndarray, (B, N+1, d) backward bridge trajectories,
                   "scaled_brownians": jnp.ndarray, (B, N, d) scaled stochastic updates for computing the gradients}
        """        
        assert initial_condition.shape[-1] == terminal_condition.shape[-1] == self.d
        B = initial_condition.shape[0]                                                  # batch size
        X = initial_condition.copy()
        Z = terminal_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))                         # (t0, t1, ..., tN)
        trajectories = trajectories.at[:, self.N, :].set(Z)                             # (Z*(t0) = X(tN))
        scaled_brownians = jnp.zeros(shape=(B, self.N, self.d))                         # (t0, t1, ..., tN-1)
        for reverse_t_idx in range(self.N, 0, -1):                                      # (N, N-1, ..., 1)
            reverse_t_now = self.ts[reverse_t_idx]                                      # (tN, tN-1..., t1)
            reverse_t_next = self.ts[reverse_t_idx-1]                                   # (tN-1, tN-2, ..., t0)
            dt = reverse_t_now - reverse_t_next                                         # (tN-tN-1, tN-1-tN-2, ..., t1-t0)
            self.rng, _ = jax.random.split(self.rng)
            score_p = eval_score(state=score_p_state, x=Z, t=reverse_t_now)
            drift = (-self.f(Z, reverse_t_now) + sb_multi(self.Sigma(Z, reverse_t_now), score_p) + self.div_Sigma(Z, reverse_t_now)) * dt

            if reverse_t_idx > 1:
                brownian = jnp.sqrt(dt) * jax.random.normal(self.rng, shape=(B, self.d))
                diffusion = sb_multi(self.g(Z, reverse_t_now), brownian)
                Z = Z + drift + diffusion       # Euler-Maruyama
                scaled_brownian = - sb_multi(self.inv_Sigma(Z, reverse_t_now) / dt, diffusion)
                scaled_brownians = scaled_brownians.at[:, reverse_t_idx-1, :].set(scaled_brownian)   # (dW(tN), dW(tN-1), ..., dW(t1))
                trajectories = trajectories.at[:, reverse_t_idx-1, :].set(Z)        # (Z*(t1), Z*(t2), ..., Z*(tN-1)) = (X(tN-1), X(tN-2), ..., X(t0))
            else:
                trajectories = trajectories.at[:, 0, :].set(X)                      # end point constraint: Z*(tN) = X(0)
                scaled_brownian = - sb_multi(self.inv_Sigma(Z, reverse_t_now) / dt, (X - Z - drift))
                scaled_brownians = scaled_brownians.at[:, 0, :].set(scaled_brownian) # (dW(tN), dW(tN-1), ..., dW(t1))
        return {"trajectories": trajectories, "scaled_brownians": scaled_brownians}
    
    def simulate_forward_bridge(self, 
                                score_p_state: TrainState,
                                score_p_star_state: TrainState,
                                initial_condition: jnp.ndarray, 
                                terminal_condition: jnp.ndarray) -> jnp.ndarray:
        """ Simulate the forward bridge process (X*(t)) which is the "backward of backward":
            dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

        Args:
            score_p_state (TrainState): s_{theta}(t, x)
            score_p_star_state (TrainState): s*_{theta}(t, x)
            initial_condition (jnp.ndarray): X*(0)
            terminal_condition (jnp.ndarray): X*(T)

        Returns:
            dict: {"trajectories": jnp.ndarray, (B, N+1, d) forward bridge trajectories,
                   "scaled_brownians": jnp.ndarray, (B, N, d) scaled stochastic updates for computing the gradients}
        """        
        assert initial_condition.shape[-1] == terminal_condition.shape[-1] == self.d

        B = initial_condition.shape[0]      # batch size
        X = initial_condition.copy()
        Z = terminal_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))
        trajectories = trajectories.at[:, 0, :].set(X)
        for t_idx in range(self.N-1):                                           # (0, 1, ..., N-2)
            if t_idx == 0:
                t_now = 0.5 * self.ts[t_idx+1]                                         # (t0)
            else:
                t_now = self.ts[t_idx]                                              # (t0, t1, ..., tN-2)
            t_next = self.ts[t_idx+1]                                           # (t1, t2, ..., tN-1)
            dt = t_next - t_now                                                 # (t1-t0, t2-t1, ..., tN-1-tN-2)
            self.rng, _ = jax.random.split(self.rng)
            score_p = eval_score(state=score_p_state, x=X, t=t_now)
            score_p_star = eval_score(state=score_p_star_state, x=X, t=t_now)
            score_h = score_p_star - score_p

            drift = (self.f(X, t_now) + sb_multi(self.Sigma(X, t_now), score_h)) * dt
            brownian = jnp.sqrt(dt) * jax.random.normal(self.rng, shape=(B, self.d))
            diffusion = sb_multi(self.g(X, t_now), brownian)
            X = X + drift + diffusion
            trajectories = trajectories.at[:, t_idx+1, :].set(X)            # (X*(t1), X*(t2), ..., X*(tN-1))

        trajectories = trajectories.at[:, self.N, :].set(Z)                 # end point constraint: X*(tN) = Z
        return {"trajectories": trajectories, "scaled_brownians": None}
    
    # def get_p_gradient(self,
    #                    forward_trajectories: jnp.ndarray,
    #                    brownian_increments: jnp.ndarray) -> jnp.ndarray:
    #     """ Compute g(t_{m-1}, X_{m-1}, t_m, X_m) for eq. (8)

    #     Args:
    #         foward_trajectories (jnp.ndarray): (B, N+1, d) forward non-bridge trajectories
    #         brownian_increments (jnp.ndarray): (B, N, d) brownian increments for computing the gradients. (* not used in this method)

    #     Returns:
    #         jnp.ndarray: (B, N, d) g(t_{m-1}, X_{m-1}, t_m, X_m)
    #     """        
    #     assert forward_trajectories.shape[-1] == self.d
    #     B = forward_trajectories.shape[0]      # batch size
    #     gradients = jnp.zeros(shape=(B, self.N, self.d))
    #     for t_idx in range(self.N-1):                                    # (0, 1, ..., N-2)
    #         X_m_minus_1 = forward_trajectories[:, t_idx, :]              # previous step for forward process 
    #         X_m = forward_trajectories[:, t_idx+1, :]                    # current step for forward process
    #         t_m_minus_1 = self.ts[t_idx]
    #         t_m = self.ts[t_idx+1]
    #         dt = t_m - t_m_minus_1
    #         gradient = -(X_m - X_m_minus_1 - dt * self.f(X_m_minus_1, t_m_minus_1)) / dt
    #         gradient = sb_multi(self.inv_Sigma(X_m_minus_1, t_m_minus_1), gradient)
    #         gradients = gradients.at[:, t_idx, :].set(gradient)
    #     return gradients
    
    def get_p_gradient(self,
                       forward_trajectories: jnp.ndarray,
                       scaled_brownians: jnp.ndarray,
                       epsilon: float=0.0) -> jnp.ndarray:
        """ Compute g(t_{m-1}, X_{m-1}, t_m, X_m) using the new expression for eq. (8):
            g(t_{m-1}, X_{m-1}, t_m, X_m) = - (\Sigma(t_{m-1}, X_{m-1}) * \delta t)^{-1} * \sigma(t_{m-1}, X_{m-1}) * \delta W(t_{m-1}, X_{m-1})

        Args:
            forward_trajectories (jnp.ndarray): (B, N+1, d) forward non-bridge trajectories.
            scaled_brownians (jnp.ndarray): (B, N, d) scaled stochastic updates for computing the gradients.
            epsilon (float, optional): a magical weight to enforce the initial constraint. Defaults to 0.0.

        Returns:
            jnp.ndarray: (B, N, d) g(t_{m-1}, X_{m-1}, t_m, X_m)
        """
        assert forward_trajectories.shape[-1] == self.d
        B = forward_trajectories.shape[0]      # batch size
        X0 = forward_trajectories[:, 0, :]     # initial condition
        gradients = jnp.zeros(shape=(B, self.N, self.d))
        for t_idx in range(self.N):                                      # (0, 1, ..., N-2)
            X_m_minus_1 = forward_trajectories[:, t_idx, :]              # previous step for forward process 
            X_m = forward_trajectories[:, t_idx+1, :]                    # current step for forward process
            if t_idx == self.N - 1:
                t_m = 0.75 * self.ts[t_idx+1] + 0.25 * self.ts[t_idx]     # (t0, t1, ..., tN-1)
            else:
                t_m = self.ts[t_idx+1]
            t_m_minus_1 = self.ts[t_idx]
            scaled_brownian = scaled_brownians[:, t_idx, :]
            additional_constraint = epsilon * sb_multi(self.inv_Sigma(X_m_minus_1, t_m_minus_1), (X0 - X_m)) / (self.T - t_m)
            gradient = scaled_brownian - additional_constraint
            gradients = gradients.at[:, t_idx, :].set(gradient)
        return gradients 
        
    # def get_p_star_gradient(self,
    #                         backward_trajectories: jnp.ndarray,
    #                         brownian_increments: jnp.ndarray) -> jnp.ndarray:
    #     """ Compute g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m) for eq. (18)

    #     Args:
    #         backward_trajectories (jnp.ndarray): (B, N+1, d) backward bridge trajectories
    #         brownian_increments (jnp.ndarray): (B, N, d) brownian increments for computing the gradients. (not used in this method)

    #     Returns:
    #         jnp.ndarray: (B, N, d) g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m)
    #     """        
    #     assert backward_trajectories.shape[-1] == self.d
    #     B = backward_trajectories.shape[0]      # batch size
    #     gradients = jnp.zeros(shape=(B, self.N, self.d))
    #     for reverse_t_idx in range(self.N, 0, -1):                              # (N, N-1, ..., 1)
    #         Z_m_minus_1 = backward_trajectories[:, reverse_t_idx, :]            # previous step for backward prcess, i.e. next step for forward process 
    #         Z_m = backward_trajectories[:, reverse_t_idx-1, :]                  # current step for backward process, also for the forward process
    #         reverse_t_m_minus_1 = self.ts[reverse_t_idx]
    #         reverse_t_m = self.ts[reverse_t_idx-1]
    #         dt = reverse_t_m_minus_1 - reverse_t_m
    #         gradient = -(Z_m - Z_m_minus_1 - dt * self.f(Z_m_minus_1, reverse_t_m_minus_1)) / dt
    #         gradient = sb_multi(self.inv_Sigma(Z_m_minus_1, reverse_t_m_minus_1), gradient)
    #         gradients = gradients.at[:, reverse_t_idx-1, :].set(gradient)
    #     return gradients
    
    def get_p_star_gradient(self,
                            backward_trajectories: jnp.ndarray,
                            scaled_brownians: jnp.ndarray,
                            epsilon: float=0.0) -> jnp.ndarray:
        """ Compute g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m) for eq. (18), but use the same trick as in get_p_gradient

        Args:
            backward_trajectories (jnp.ndarray): (B, N+1, d) backward bridge trajectories
            scaled_brownians (jnp.ndarray): (B, N, d) scaled stochastic updates for computing the gradients.
            epsilon (float, optional): a magical weight to enforce the initial constraint. Defaults to 1e-4.

        Returns:
            jnp.ndarray: (B, N, d) g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m)
        """        
        assert backward_trajectories.shape[-1] == self.d
        B = backward_trajectories.shape[0]      # batch size
        XT = backward_trajectories[:, -1, :]    # terminal condition
        gradients = jnp.zeros(shape=(B, self.N, self.d))
        for reverse_t_idx in range(self.N, 0, -1):                              # (N, N-1, ..., 2)
            Z_m_minus_1 = backward_trajectories[:, reverse_t_idx, :]            # previous step for backward prcess, i.e. next step for forward process 
            Z_m = backward_trajectories[:, reverse_t_idx-1, :]                  # current step for backward process, also for the forward process
            reverse_t_m_minus_1 = self.ts[reverse_t_idx]
            if reverse_t_idx == 1:
                reverse_t_m = 0.25 * (self.ts[reverse_t_idx] - self.ts[reverse_t_idx-1])
            else:
                reverse_t_m = self.ts[reverse_t_idx-1]
            dt = reverse_t_m_minus_1 - reverse_t_m
            scaled_brownian = scaled_brownians[:, reverse_t_idx-1, :]
            additional_constraint = epsilon * sb_multi(self.inv_Sigma(Z_m_minus_1, reverse_t_m_minus_1), (XT - Z_m)) / reverse_t_m
            gradient = scaled_brownian - additional_constraint
            gradients = gradients.at[:, reverse_t_idx-1, :].set(gradient)
        return gradients
    
    def get_trajectories_generator(self,
                                   batch_size: int,
                                   process_type: str,
                                   initial_condition: jnp.ndarray,
                                   terminal_condition: jnp.ndarray,
                                   score_p_state: TrainState,
                                   score_p_star_state: TrainState) -> callable:
        assert process_type in ['forward', 'backward_bridge', 'forward_bridge']
        assert initial_condition.shape[-1] == self.d
        def generator():
            initial_conditions = jnp.tile(initial_condition, reps=(batch_size, 1))
            terminal_conditions = jnp.tile(terminal_condition, reps=(batch_size, 1)) if terminal_condition is not None else None
            while True:
                if process_type == 'forward':
                    histories = self.simulate_forward_process(initial_conditions)
                elif process_type == 'backward_bridge':
                    histories = self.simulate_backward_bridge(score_p_state, 
                                                              initial_conditions, 
                                                              terminal_conditions)
                elif process_type == 'forward_bridge':
                    histories = self.simulate_forward_bridge(score_p_state,
                                                             score_p_star_state, 
                                                             initial_conditions, 
                                                             terminal_conditions)
                yield (histories["trajectories"], histories["scaled_brownians"])
        return generator
        
    def learn_p_score(self,
                      initial_condition: jnp.ndarray,
                      setup_params: dict):
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_net = ScoreNet(**net_params)
        data_generator = self.get_trajectories_generator(batch_size=training_params["batch_size"],
                                                         process_type='forward',
                                                         initial_condition=initial_condition,
                                                         terminal_condition=None,
                                                         score_p_state=None,
                                                         score_p_star_state=None)
        dataset = get_iterable_dataset(generator=data_generator, 
                                       dtype=(tf.float32, tf.float32), 
                                       shape=[(training_params["batch_size"], self.N+1, self.d),
                                              (training_params["batch_size"], self.N, self.d)])

        @jax.jit
        def train_step(state: TrainState,
                       batch: tuple):
            trajectories, scaled_brownians = batch
            ts = flatten_batch(unsqueeze(jnp.tile(self.ts[1:], reps=(training_params["batch_size"], 1)), axis=-1))        # (B*N, 1)
            score_p_gradients = self.get_p_gradient(trajectories, scaled_brownians)                                       # (B, N, d)
            score_p_gradients = flatten_batch(score_p_gradients)                                              # (B*N, d)
            trajectories = flatten_batch(trajectories[:, 1:, :])                                                    # (B*N, d)
            def loss_fn(params):
                score_p_est, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, 
                                                      x=trajectories, 
                                                      t=ts, 
                                                      train=True, 
                                                      mutable=['batch_stats'])                                                # (B*N, d)
                loss = jnp.mean(jnp.square((score_p_est - score_p_gradients)))                       
                return loss, updates
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        
        state = create_train_state(score_p_net, 
                                   self.rng, 
                                   training_params["learning_rate"], 
                                   [(training_params["batch_size"], self.d), (training_params["batch_size"], 1)])
        pbar = tqdm(range(training_params["num_epochs"]), desc="Training", leave=True, unit="epoch", total=training_params["num_epochs"])
        for i in pbar:

            for _ in range(training_params["num_batches_per_epoch"]):
                batch = next(dataset)
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i+1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state

    def learn_p_star_score(self,
                           initial_condition: jnp.ndarray,
                           terminal_condition: jnp.ndarray,
                           score_p_state: TrainState,
                           setup_params: dict):
        assert "network" in setup_params.keys() and "training" in setup_params.keys()
        net_params = setup_params["network"]
        training_params = setup_params["training"]
        score_p_star_net = ScoreNet(**net_params)

        data_generator = self.get_trajectories_generator(batch_size=training_params["batch_size"],
                                                         process_type='backward_bridge',
                                                         initial_condition=initial_condition,
                                                         terminal_condition=terminal_condition,
                                                         score_p_state=score_p_state,
                                                         score_p_star_state=None)
        dataset = get_iterable_dataset(generator=data_generator, 
                                       dtype=(tf.float32, tf.float32), 
                                       shape=[(training_params["batch_size"], self.N+1, self.d),
                                              (training_params["batch_size"], self.N, self.d)])

        @jax.jit
        def train_step(state: TrainState, 
                       batch: tuple):
            trajectories, scaled_brownians = batch
            ts = flatten_batch(unsqueeze(jnp.tile(self.ts[:-1], reps=(training_params["batch_size"], 1)), axis=-1))        # (B*N, 1)
            score_p_star_gradients = self.get_p_star_gradient(trajectories, scaled_brownians)                                             # (B, N, d)
            score_p_star_gradients = flatten_batch(score_p_star_gradients)                                                    # (B*N, d)
            trajectories = flatten_batch(trajectories[:, :-1, :])                                                      # (B*N, d)
            def loss_fn(params):
                score_p_star_est, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, 
                                                             x=trajectories, 
                                                             t=ts, 
                                                             train=True, 
                                                             mutable=['batch_stats'])                                                # (B*N, d)
                loss = jnp.mean(jnp.square((score_p_star_est - score_p_star_gradients)))                           
                return loss, updates
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        
        state = create_train_state(score_p_star_net, 
                                   self.rng, 
                                   training_params["learning_rate"], 
                                   [(training_params["batch_size"], self.d), (training_params["batch_size"], 1)])
        pbar = tqdm(range(training_params["num_epochs"]), desc="Training", leave=True, unit="epoch", total=training_params["num_epochs"])
        for i in pbar:

            for _ in range(training_params["num_batches_per_epoch"]):
                batch = next(dataset)
                state = train_step(state, batch)
            pbar.set_postfix(Epoch=i+1, loss=f"{state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state