import functools
from collections import namedtuple
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
from einops import rearrange, repeat
from tqdm import tqdm

from sdebridge import sdes
from sdebridge.utils import create_train_state, get_iterable_dataset, weighted_norm_square


def trajectory_generator(
    sde: sdes.SDE,
    key: jax.Array,
    batch_size: int,
    x0: jnp.ndarray,
) -> Callable:
    initial_vals = jnp.tile(x0, reps=(batch_size, 1, 1))

    def generator():
        subkey = key
        while True:
            step_key, subkey = jax.random.split(subkey)
            # trajs = sde.simulate_trajectories(initial_val, key=key, num_batches=batch_size)
            # grads, covs = sde.grad_and_covariance(trajs)
            trajs, grads, covs = euler_and_grad_and_cov(sde, initial_vals, key)
            yield trajs, grads, covs,

    return generator


def learn_p_score(
    sde: sdes.SDE,
    initial_val: jax.Array,
    key: jax.Array,
    *,
    batch_size: int,
    load_size: int,
    learning_rate: int,
    warmup_steps: int,
    num_epochs: int,
    net: nn.Module,
    network_params: dict = None,
):
    ts = rearrange(
        repeat(sde.ts[1:], "n -> b n", b=batch_size),
        "b n -> (b n) 1",
        b=batch_size,
    )
    gen = trajectory_generator(sde, key, load_size, initial_val)
    return learn_score(
        sde,
        gen,
        key,
        ts,
        batch_size=batch_size,
        load_size=load_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_epochs=num_epochs,
        net=net,
        network_params=network_params,
    )


def learn_p_star_score(
    forward_sde: sdes.SDE,
    initial_val: jax.Array,
    key: jax.Array,
    score_p: Callable,
    *,
    batch_size: int,
    load_size: int,
    learning_rate: int,
    warmup_steps: int,
    num_epochs: int,
    net: nn.Module,
    network_params: dict = None,
):
    ts = rearrange(
        repeat(forward_sde.T - forward_sde.ts[:-1], "n -> b n", b=batch_size),
        # !!! the backward trajectories are in the reverse order, so we need inverted time series.
        "b n -> (b n) 1",
        b=batch_size,
    )
    reverse_sde = sdes.reverse(forward_sde, score_p)
    gen = trajectory_generator(reverse_sde, key, load_size, initial_val)

    return learn_score(
        forward_sde,
        gen,
        key,
        ts,
        batch_size=batch_size,
        load_size=load_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_epochs=num_epochs,
        net=net,
        network_params=network_params,
    )


def learn_score(
    sde: sdes.SDE,
    generator: Callable,
    key: jax.Array,
    ts: jax.Array,
    *,
    batch_size: int,
    load_size: int,
    learning_rate: int,
    warmup_steps: int,
    num_epochs: int,
    net: nn.Module,
    network_params: dict = None,
):
    net_params = network_params
    num_batches_per_epoch = int(load_size / batch_size)
    score_net = net(**net_params)

    _, network_key = jax.random.split(key)

    iter_dataset = get_iterable_dataset(
        generator=generator,
        dtype=(tf.float64, tf.float64, tf.float64),
        shape=[
            (
                load_size,
                sde.N - 1,
                2,
                sde.n_bases,
                sde.dim,
            ),  # todo: don't hard code the 2 here (should also work for less axes)
            (load_size, sde.N - 1, 2, sde.n_bases, sde.dim),
            (load_size, sde.N - 1, 2, sde.n_bases, sde.n_bases),
        ],
    )

    @jax.jit
    def train_step(state, batch: tuple):
        trajs, grads, covs = batch  # (B, N, n_bases, 2)
        b = trajs.shape[0]
        n = trajs.shape[1]

        trajs = trajs.reshape((b * n, *trajs.shape[2:]))
        grads = grads.reshape((b * n, *grads.shape[2:]))
        covs = covs.reshape((b * n, *covs.shape[2:]))

        # trajs = rearrange(trajs, "b n d1 d2 -> (b n) d1 d2")  # (B*N, n_bases, 2)
        # grads = rearrange(grads, "b n d1 d2 -> (b n) d1 d2")  # (B*N, n_bases, 2)
        # covs = rearrange(covs, "b n d1 d2 -> (b n) d1 d2")  # (B*N, n_bases, n_bases)

        def loss_fn(params) -> tuple:
            score, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                x=trajs,
                t=ts,
                train=True,
                mutable=["batch_stats"],
            )  # (B*N, 2*n_bases)
            loss = weighted_norm_square(x=score - grads, covariance=covs)  # (B*N, d) -> (B*N, )
            loss = 0.5 * sde.dt * jnp.mean(loss, axis=0)
            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates["batch_stats"])
        step_key, _ = jax.random.split(state.key)
        state = state.replace(key=step_key)

        return state, loss

    state = create_train_state(
        model=score_net,
        key=network_key,
        input_shapes=[
            (batch_size, sde.n_bases, sde.dim),
            (batch_size, 1),
        ],
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_epochs * num_batches_per_epoch,
    )
    pbar = tqdm(
        range(num_epochs),
        desc="Training",
        leave=True,
        unit="epoch",
        total=num_epochs,
    )
    for i in pbar:
        total_loss = 0
        load = next(iter_dataset)
        for b in range(num_batches_per_epoch):
            tmp1, tmp2, tmp3 = load
            batch = (
                tmp1[b * batch_size : (b + 1) * batch_size],
                tmp2[b * batch_size : (b + 1) * batch_size],
                tmp3[b * batch_size : (b + 1) * batch_size],
            )
            state, loss = train_step(state, batch)
            total_loss += loss
        epoch_loss = total_loss / num_batches_per_epoch
        pbar.set_postfix(Epoch=i + 1, loss=f"{epoch_loss:.4f}")

    return state


def batch_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Batch matrix multiplication"""
    return jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)(A, B)


def euler_and_grad_and_cov(
    sde: sdes.SDE,
    initial_vals: jnp.ndarray,
    rng_key: jax.Array,
) -> tuple:
    """Euler-Maruyama solver for SDEs

    initial_vals: (B, 2*N), complex64
    terminal_vals: (B, 2*N), complex64
    """
    b = initial_vals.shape[0]
    state = namedtuple("state", ["x", "grads", "covs", "key"])
    init_state = state(
        x=initial_vals,
        grads=jnp.empty_like(initial_vals),
        covs=jnp.empty((b, sde.n_bases, sde.n_bases)),
        key=rng_key,
    )

    def euler_maruyama_step(s: state, time: jnp.ndarray) -> tuple:
        """Euler-Maruyama step, NOTE: all the calculations are over batches"""
        time = jnp.expand_dims(time, axis=-1)
        time = jnp.tile(time, (b, 1))
        step_key, _ = jax.random.split(s.key)
        drift_ = jax.vmap(sde.drift, in_axes=(0, 0))(s.x, time)  # (b, 2*n_bases)

        eps_ = jax.random.normal(step_key, shape=(b, *sde.bm_shape))
        diffusion_ = jax.vmap(sde.diffusion, in_axes=(0, None))(s.x, time)  # (B, 2*n_bases, 2*n_grid**2)
        diffusion_step = jnp.sqrt(sde.dt) * diffusion_ @ eps_

        cov_ = jax.vmap(sdes.cov, in_axes=(None, 0, None))(sde, s.x, time)  # (B, 2*n_bases, 2*n_bases)
        inv_cov = jax.vmap(partial(jnp.linalg.pinv, hermitian=True, rcond=None))(cov_)  # (B, 2*n_bases, 2*n_bases)

        grads = -1 / sde.dt * batch_matmul(inv_cov, diffusion_step)  # (B, 2*n_bases)

        xnew = s.x + drift_ * sde.dt + diffusion_step  # (B, 2*n_bases)
        new_state = state(
            x=xnew,
            grads=grads,
            covs=cov_,
            key=step_key,
        )
        return new_state, (
            s.x,
            s.grads,
            s.covs,
            s.key,
        )

    _, (trajectories, gradients, covariances, step_keys) = jax.lax.scan(
        euler_maruyama_step,
        init=init_state,
        xs=(sde.ts[:-1]),
        length=sde.N - 1,
    )
    trajectories = trajectories.swapaxes(0, 1)
    gradients = gradients.swapaxes(0, 1)
    covariances = covariances.swapaxes(0, 1)

    return trajectories, gradients, covariances
