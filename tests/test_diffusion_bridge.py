import jax
import jax.numpy as jnp
import pytest

from sdebridge.diffusion_bridge import DiffusionBridge


@pytest.fixture
def brownian_1d():
    def drift(*args):
        return jnp.zeros(1)

    def diffusion(*args):
        return jnp.identity(1)

    def brownian_transition_1d(time, value, start_val=0, start_time=0):
        return -(value - start_val) / (time - start_time)

    return DiffusionBridge(
        drift=drift,
        diffusion=diffusion,
        dim=1,
        end_time=1,
        num_steps=10,
        rng=jax.random.PRNGKey(1),
        true_score_transition=brownian_transition_1d,
    )


def test_simulate_forward_process2(brownian_1d):
    process = brownian_1d.simulate_forward_process2(jnp.asarray([0], dtype=jnp.float32))
    trajectories = process["trajectories"]
    increments = process["brownian_increments"]
    assert trajectories.shape == (1, 11, 1)
    assert trajectories[0, 0, 0] == 0
    assert trajectories[0, -1, 0] == jnp.sum(increments, axis=None)
    assert trajectories[0, :, 0].all() < 100
    assert not jnp.isinf(trajectories[0, :, 0]).any()


def test_simulate_backward_bridge(brownian_1d):
    process = brownian_1d.simulate_backward_bridge(
        initial_condition=jnp.asarray([[0.0]]),
        terminal_condition=jnp.asarray([[0.0]]),
        using_true_score=True,
    )
    trajectories = process["trajectories"]
    increments = process["brownian_increments"]
    assert trajectories.shape == (1, 11, 1)
    assert trajectories[0, 0, 0] == 0
    assert trajectories[0, -1, 0] == 0
    assert trajectories[0, :, 0].all() < 100
    assert not jnp.isinf(trajectories[0, :, 0]).any()


def test_simulate_backward_bridge2(brownian_1d):
    process = brownian_1d.simulate_backward_bridge2(
        initial_condition=jnp.asarray([[0.0]]),
        terminal_condition=jnp.asarray([[0.0]]),
        using_true_score=True,
    )
    trajectories = process["trajectories"]
    increments = process["brownian_increments"]
    assert trajectories.shape == (1, 11, 1)
    print(trajectories)
    assert trajectories[0, 0, 0] == 0
    assert trajectories[0, -1, 0] == 0
    assert trajectories[0, :, 0].all() < 100
    assert not jnp.isinf(trajectories[0, :, 0]).any()


def test_forward(brownian_1d):
    process1 = brownian_1d.simulate_forward_process(jnp.asarray([0], dtype=jnp.float32))
    traj1 = process1["trajectories"]
    increment1 = process1["brownian_increments"]
    process2 = brownian_1d.simulate_forward_process(jnp.asarray([0], dtype=jnp.float32))
    traj2 = process2["trajectories"]
    increment2 = process2["brownian_increments"]
    assert jnp.array_equal(traj1, traj2)
    assert jnp.array_equal(increment1, increment2)


def test_backward(brownian_1d):
    process1 = brownian_1d.simulate_backward_bridge(
        initial_condition=jnp.asarray([[0.0]]),
        terminal_condition=jnp.asarray([[0.0]]),
        using_true_score=True,
    )
    process2 = brownian_1d.simulate_backward_bridge2(
        initial_condition=jnp.asarray([[0.0]]),
        terminal_condition=jnp.asarray([[0.0]]),
        using_true_score=True,
    )
    traj1 = process1["trajectories"]
    increment1 = process1["brownian_increments"]
    traj2 = process2["trajectories"]
    increment2 = process2["brownian_increments"]
    assert jnp.array_equal(traj1, traj2)
    assert jnp.array_equal(increment1, increment2)
