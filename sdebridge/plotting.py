from functools import partial

from .setup import *


def plot_2d_vector_field(
    X: callable,
    X_ref: callable,
    xs: jnp.ndarray,
    ts: jnp.ndarray,
    suptitle: str,
    scale: float = None,
    **kwargs,
):
    xm, ym = jnp.meshgrid(xs, xs)
    x = jnp.stack([xm.flatten(), ym.flatten()], axis=-1)
    fig, ax = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4))
    X_ref = partial(X_ref, **kwargs) if X_ref is not None else None
    for i, t in enumerate(ts):
        vector_field = X(x, t) if X is not None else None
        vector_field_ref = X_ref(x, t) if X_ref is not None else None
        if vector_field is not None:
            u = vector_field[:, 0].reshape(xm.shape)
            v = vector_field[:, 1].reshape(xm.shape)
            ax[i].quiver(xm, ym, u, v, color="b", scale=scale)

        if vector_field_ref is not None:
            u_ref = vector_field_ref[:, 0].reshape(xm.shape)
            v_ref = vector_field_ref[:, 1].reshape(xm.shape)
            ax[i].quiver(xm, ym, u_ref, v_ref, color="r", scale=scale)

        ax[i].set_title(f"t = {t:.1f}")
    fig.suptitle(suptitle)
    plt.show()


def plot_trajectories(trajectories: jnp.ndarray, title: str, **kwargs):
    colormap = plt.cm.get_cmap("jet")
    assert len(trajectories.shape) == 3
    num_trajectories = trajectories.shape[0]
    dim = trajectories.shape[2]
    colors = [colormap(i) for i in jnp.linspace(0, 1, num_trajectories)]
    plt.figure(figsize=(5, 5))
    for i in range(num_trajectories):
        for j in range(dim // 2):
            plt.plot(
                trajectories[i, :, 2 * j],
                trajectories[i, :, 2 * j + 1],
                color=colors[i],
                zorder=1,
                alpha=0.7,
                **kwargs,
            )
            plt.scatter(
                trajectories[i, 0, 2 * j],
                trajectories[i, 0, 2 * j + 1],
                color="b",
                marker="o",
                edgecolors="k",
                zorder=2,
            )
            plt.scatter(
                trajectories[i, -1, 2 * j],
                trajectories[i, -1, 2 * j + 1],
                color=colors[i],
                marker="D",
                edgecolors="k",
                zorder=2,
            )
    plt.axis("equal")
    plt.title(title)


def plot_single_trajectory(trajectory: jnp.ndarray, title: str, **kwargs):
    colormap = plt.cm.get_cmap("jet")
    assert len(trajectory.shape) == 2
    dim = trajectory.shape[-1]
    colors = [colormap(i) for i in jnp.linspace(0, 1, dim // 2)]
    plt.figure(figsize=(5, 5))
    for i in range(dim // 2):
        plt.plot(
            trajectory[:, 2 * i],
            trajectory[:, 2 * i + 1],
            color=colors[i],
            zorder=1,
            alpha=0.5,
            **kwargs,
        )
        plt.scatter(
            trajectory[0, 2 * i],
            trajectory[0, 2 * i + 1],
            color=colors[i],
            marker="o",
            edgecolors="k",
            zorder=2,
        )
        plt.scatter(
            trajectory[-1, 2 * i],
            trajectory[-1, 2 * i + 1],
            color=colors[i],
            marker="D",
            edgecolors="k",
            zorder=2,
        )
    plt.axis("equal")
    plt.title(title)
