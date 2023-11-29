import jax
import jax.numpy as jnp

from .utils import *

class DiffusionBridge:
    def __init__(self,
                 drift: callable,
                 diffusion: callable,
                 dim: int,
                 end_time: float,
                 num_steps: int,
                 true_score_transition: callable=None,
                 true_score_h: callable=None,
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
        self.true_score_transition = true_score_transition
        self.true_score_h = true_score_h

    def simulate_forward_process(self, 
                                 initial_condition: jnp.ndarray) -> jnp.ndarray:
        """ Simulate the forward non-bridge process (X(t)):
            dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)

        Args:
            initial_condition (jnp.ndarray): X(0) 

        Returns:
            jnp.ndarray: (B, N+1, d) forward non-bridge trajectories
        """        
        assert initial_condition.shape[-1] == self.d
        B = initial_condition.shape[0]                                  # batch size
        X = initial_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))         # (t0, t1, ..., tN)
        trajectories = trajectories.at[:, 0, :].set(X)                  # (X(t0))                
        for t_idx in range(self.N):                                     # (0, 1, ... , N-1)
            t_now = self.ts[t_idx]                                      # (t0, t1, ..., tN-1)
            t_next = self.ts[t_idx+1]                                   # (t1, t2, ..., tN)
            dt = t_next - t_now                                         # (t1-t0, t2-t1, ..., tN-tN-1)
            self.rng, _ = jax.random.split(self.rng)
            drift = dt * self.f(X, t_now) 
            diffusion = jnp.sqrt(dt) * sb_multi(self.g(X, t_now), jax.random.normal(self.rng, shape=(B, self.d)))
            X = X + drift + diffusion                                   # Euler-Maruyama
            trajectories = trajectories.at[:, t_idx + 1, :].set(X)      # (X(t1), X(t2), ..., X(tN))
        return trajectories
    
    def simulate_backward_bridge(self, 
                                 score_transition_state: TrainState,
                                 initial_condition: jnp.ndarray, 
                                 terminal_condition: jnp.ndarray, 
                                 using_true_score: bool) -> jnp.ndarray:
        """ Simulate the backward bridge process (Z*(t)):
            dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        Args:
            score_transition_state (TrainState): s_{theta}(t, x)
            initial_condition (jnp.ndarray): Z*(0) = X(T-t)
            terminal_condition (jnp.ndarray): Z*(T) = X(0)
            using_true_score (bool): If True, use the predefined score transition function.

        Returns:
            jnp.ndarray: (B, N+1, d) backward bridge trajectories
        """        
        assert initial_condition.shape[-1] == terminal_condition.shape[-1] == self.d
        if using_true_score:
            assert self.true_score_transition is not None
        B = initial_condition.shape[0]                                                  # batch size
        Z = initial_condition.copy()
        X = terminal_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))                         # (t0, t1, ..., tN)
        trajectories = trajectories.at[:, self.N, :].set(Z)                             # (Z*(t0) = X(tN))
        for reverse_t_idx in range(self.N, 0, -1):                                      # (N, N-1, ..., 1)
            reverse_t_now = self.ts[reverse_t_idx]                                      # (tN, tN-1..., t1)
            reverse_t_next = self.ts[reverse_t_idx-1]                                   # (tN-1, tN-2, ..., t0)
            dt = reverse_t_now - reverse_t_next                                         # (tN-tN-1, tN-1-tN-2, ..., t1-t0)
            self.rng, _ = jax.random.split(self.rng)
            if using_true_score:
                score_transition = self.true_score_transition(Z, X, reverse_t_now)
            else:
                score_transition = eval_score(state=score_transition_state, x=Z, t=reverse_t_now)
            drift = (-self.f(Z, reverse_t_now) + sb_multi(self.Sigma(Z, reverse_t_now), score_transition) + self.div_Sigma(Z, reverse_t_now)) * dt
            diffusion = jnp.sqrt(dt) * sb_multi(self.g(Z, reverse_t_now), jax.random.normal(self.rng, shape=(B, self.d)))
            Z = Z + drift + diffusion       # Euler-Maruyama
            trajectories = trajectories.at[:, reverse_t_idx-1, :].set(Z)        # (Z*(t1), Z*(t2), ..., Z*(tN-1)) = (X(tN-1), X(tN-2), ..., X(t0))
        trajectories = trajectories.at[:, 0, :].set(X)                          # end point constraint: Z*(tN) = X(0)
        return trajectories
    
    def simulate_forward_bridge(self, 
                                score_transition_state: TrainState,
                                score_marginal_state: TrainState,
                                initial_condition: jnp.ndarray, 
                                terminal_condition: jnp.ndarray, 
                                using_true_score: bool) -> jnp.ndarray:
        """ Simulate the forward bridge process (X*(t)):
            dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

        Args:
            score_transition_state (TrainState): s_{theta}(t, x)
            score_marginal_state (TrainState): s*_{theta}(t, x)
            initial_condition (jnp.ndarray): X*(0)
            terminal_condition (jnp.ndarray): X*(T)
            using_true_score (bool): If True, use the predefined score h function.

        Returns:
            jnp.ndarray: (B, N+1, d) forward bridge trajectories
        """        
        assert initial_condition.shape[-1] == terminal_condition.shape[-1] == self.d
        if using_true_score:
            assert self.true_score_h is not None

        B = initial_condition.shape[0]      # batch size
        X = initial_condition.copy()
        Z = terminal_condition.copy()
        trajectories = jnp.zeros(shape=(B, self.N + 1, self.d))
        trajectories = trajectories.at[:, 0, :].set(X)
        for t_idx in range(self.N-1):                                           # (0, 1, ..., N-2)
            t_now = self.ts[t_idx]                                              # (t0, t1, ..., tN-2)
            t_next = self.ts[t_idx+1]                                           # (t1, t2, ..., tN-1)
            dt = t_next - t_now                                                 # (t1-t0, t2-t1, ..., tN-1-tN-2)
            self.rng, _ = jax.random.split(self.rng)
            if using_true_score:
                score_h = self.true_score_h(X, Z, t_now, self.T)
            else:
                score_transition = eval_score(state=score_transition_state, x=X, t=t_now)
                score_marginal = eval_score(state=score_marginal_state, x=X, t=t_now)
                score_h = score_marginal - score_transition

            drift = (self.f(X, t_now) + sb_multi(self.Sigma(X, t_now), score_h)) * dt
            diffusion =jnp.sqrt(dt) * sb_multi(self.g(X, t_now), jax.random.normal(self.rng, shape=(B, self.d)))
            X = X + drift + diffusion
            trajectories = trajectories.at[:, t_idx+1, :].set(X)            # (X*(t1), X*(t2), ..., X*(tN-1))
        trajectories = trajectories.at[:, self.N, :].set(Z)                 # end point constraint: X*(tN) = Z
        return trajectories
    
    def get_transition_gradient(self,
                                foward_trajectories: jnp.ndarray) -> jnp.ndarray:
        """ Compute g(t_{m-1}, X_{m-1}, t_m, X_m) for eq. (8)

        Args:
            foward_trajectories (jnp.ndarray): (B, N+1, d) forward non-bridge trajectories

        Returns:
            jnp.ndarray: (B, N, d) g(t_{m-1}, X_{m-1}, t_m, X_m)
        """        
        assert foward_trajectories.shape[-1] == self.d
        B = foward_trajectories.shape[0]      # batch size
        gradients = jnp.zeros(shape=(B, self.N, self.d))
        for t_idx in range(self.N-1):                                   # (0, 1, ..., N-2)
            X_m_minus_1 = foward_trajectories[:, t_idx, :]              # previous step for forward process 
            X_m = foward_trajectories[:, t_idx+1, :]                    # current step for forward process
            t_m_minus_1 = self.ts[t_idx]
            t_m = self.ts[t_idx+1]
            dt = t_m - t_m_minus_1
            gradient = -(X_m - X_m_minus_1 - dt * self.f(X_m_minus_1, t_m_minus_1)) / dt
            gradient = sb_multi(self.inv_Sigma(X_m_minus_1, t_m_minus_1), gradient)
            gradients = gradients.at[:, t_idx, :].set(gradient)
        return gradients
    
    def get_marginal_gradient(self,
                              backward_trajectories: jnp.ndarray) -> jnp.ndarray:
        """ Compute g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m) for eq. (18)

        Args:
            backward_trajectories (jnp.ndarray): (B, N+1, d) backward bridge trajectories

        Returns:
            jnp.ndarray: (B, N, d) g^(t_{m-1}, Z*_{m-1}, t_m, Z*_m)
        """        
        assert backward_trajectories.shape[-1] == self.d
        B = backward_trajectories.shape[0]      # batch size
        gradients = jnp.zeros(shape=(B, self.N, self.d))
        for reverse_t_idx in range(self.N, 0, -1):                              # (N, N-1, ..., 1)
            Z_m_minus_1 = backward_trajectories[:, reverse_t_idx, :]            # previous step for backward prcess, i.e. next step for forward process 
            Z_m = backward_trajectories[:, reverse_t_idx-1, :]                  # current step for backward process, also for the forward process
            reverse_t_m_minus_1 = self.ts[reverse_t_idx]
            reverse_t_m = self.ts[reverse_t_idx-1]
            dt = reverse_t_m_minus_1 - reverse_t_m
            gradient = -(Z_m - Z_m_minus_1 - dt * self.f(Z_m_minus_1, reverse_t_m_minus_1)) / dt
            gradient = sb_multi(self.inv_Sigma(Z_m_minus_1, reverse_t_m_minus_1), gradient)
            gradients = gradients.at[:, reverse_t_idx-1, :].set(gradient)
        return gradients
    
    def get_trajectories_generator(self,
                                   batch_size: int,
                                   process_type: str,
                                   initial_condition: jnp.ndarray,
                                   terminal_condition: jnp.ndarray,
                                   score_transition_state: TrainState,
                                   score_marginal_state: TrainState,
                                   using_true_score: bool=True) -> callable:
        assert process_type in ['forward', 'backward_bridge', 'forward_bridge']
        assert initial_condition.shape[-1] == self.d
        def generator():
            initial_conditions = jnp.tile(initial_condition, reps=(batch_size, 1))
            terminal_conditions = jnp.tile(terminal_condition, reps=(batch_size, 1)) if terminal_condition is not None else None
            while True:
                if process_type == 'forward':
                    trajectories = self.simulate_forward_process(initial_conditions)
                elif process_type == 'backward_bridge':
                    if not using_true_score:
                        assert terminal_condition is not None
                        assert score_transition_state is not None
                    else:
                        assert self.true_score_transition is not None
                    trajectories = self.simulate_backward_bridge(score_transition_state, 
                                                                 initial_conditions, 
                                                                 terminal_conditions, 
                                                                 using_true_score)
                elif process_type == 'forward_bridge':
                    if not using_true_score:
                        assert terminal_condition is not None
                        assert score_transition_state is not None
                        assert score_marginal_state is not None
                    else:
                        assert self.true_score_transition is not None
                    trajectories = self.simulate_forward_bridge(score_transition_state,
                                                                score_marginal_state, 
                                                                initial_conditions, 
                                                                terminal_conditions, 
                                                                using_true_score)
                yield trajectories
        return generator
        
    def learn_forward_transition_score(self,
                                       score_transition_net: nn.Module,
                                       initial_condition: jnp.ndarray,
                                       training_params: dict):
        data_generator = self.get_trajectories_generator(batch_size=training_params["batch_size"],
                                                         process_type='forward',
                                                         initial_condition=initial_condition,
                                                         terminal_condition=None,
                                                         score_transition_state=None,
                                                         score_marginal_state=None,
                                                         using_true_score=False)
        dataset = get_iterable_dataset(generator=data_generator, dtype=tf.float32, shape=(training_params["batch_size"], self.N+1, self.d))

        @jax.jit
        def train_step(state: TrainState, batch_trajectories: jnp.ndarray):
            batch_ts = flatten_batch(unsqueeze(jnp.tile(self.ts[1:], reps=(training_params["batch_size"], 1)), axis=-1))        # (B*N, 1)
            score_transition_gradients = self.get_transition_gradient(batch_trajectories)                                       # (B, N, d)
            score_transition_gradients = flatten_batch(score_transition_gradients)                                              # (B*N, d)
            batch_trajectories = flatten_batch(batch_trajectories[:, 1:, :])                                                    # (B*N, d)
            def loss_fn(params):
                score_transition_est, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, 
                                                               x=batch_trajectories, 
                                                               t=batch_ts, 
                                                               train=True, 
                                                               mutable=['batch_stats'])                                                # (B*N, d)
                loss = jnp.mean(jnp.square((score_transition_est - score_transition_gradients)))                       
                return loss, updates
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        
        state = create_train_state(score_transition_net, self.rng, training_params["learning_rate"], [(training_params["batch_size"], self.d), (training_params["batch_size"], 1)])
        for i in range(training_params["num_epochs"]):

            for _ in range(training_params["num_batches_per_epoch"]):
                batch_trajectories = next(dataset)
                state = train_step(state, batch_trajectories)
            print(f"Epoch {i+1} / {training_params['num_epochs']}: loss = {state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state

    def learn_marginal_score(self,
                             score_marginal_net: nn.Module,
                             initial_condition: jnp.ndarray,
                             terminal_condition: jnp.ndarray,
                             score_transition_state: TrainState,
                             training_params: dict):
        using_true_score = score_transition_state is None 
        data_generator = self.get_trajectories_generator(batch_size=training_params["batch_size"],
                                                         process_type='backward_bridge',
                                                         initial_condition=initial_condition,
                                                         terminal_condition=terminal_condition,
                                                         score_transition_state=score_transition_state,
                                                         score_marginal_state=None,
                                                         using_true_score=using_true_score)
        dataset = get_iterable_dataset(generator=data_generator, dtype=tf.float32, shape=(training_params["batch_size"], self.N+1, self.d))

        @jax.jit
        def train_step(state: TrainState, batch_trajectories: jnp.ndarray):
            batch_ts = flatten_batch(unsqueeze(jnp.tile(self.ts[:-1], reps=(training_params["batch_size"], 1)), axis=-1))        # (B*N, 1)
            score_marginal_gradients = self.get_marginal_gradient(batch_trajectories)                                             # (B, N, d)
            score_marginal_gradients = flatten_batch(score_marginal_gradients)                                                    # (B*N, d)
            batch_trajectories = flatten_batch(batch_trajectories[:, :-1, :])                                                      # (B*N, d)
            def loss_fn(params):
                score_marginal_est, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, 
                                                             x=batch_trajectories, 
                                                             t=batch_ts, 
                                                             train=True, 
                                                             mutable=['batch_stats'])                                                # (B*N, d)
                loss = jnp.mean(jnp.square((score_marginal_est - score_marginal_gradients)))                           
                return loss, updates
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])

            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        
        state = create_train_state(score_marginal_net, self.rng, training_params["learning_rate"], [(training_params["batch_size"], self.d), (training_params["batch_size"], 1)])
        for i in range(training_params["num_epochs"]):

            for _ in range(training_params["num_batches_per_epoch"]):
                batch_trajectories = next(dataset)
                state = train_step(state, batch_trajectories)
            print(f"Epoch {i+1} / {training_params['num_epochs']}: loss = {state.metrics.compute()['loss']:.4f}")
            state = state.replace(metrics=state.metrics.empty())

        return state