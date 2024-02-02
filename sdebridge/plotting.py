from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from tueplots import axes, bundles, cycler, figsizes, fonts
from tueplots.constants.color import palettes

from .setup import *


def set_style(column="half", nrows=1, ncols=1):
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(
        bundles.icml2024(column=column, nrows=nrows, ncols=ncols, usetex=True)
    )
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(axes.spines(top=False, right=False))


def set_style2(nrows, ncols):
    plt.rcParams.update(figsizes.icml2024_full(nrows=nrows, ncols=ncols))
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))


def plot_butterfly_traj_pts(traj, sample_idx, ax, cmap_name="viridis"):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0, 1, traj.shape[1]))
    t = jnp.linspace(0, 1, traj.shape[1])
    shift = jnp.linspace(0, 0.0, traj.shape[1])
    traj = add_start_to_end(traj)
    ax.plot(
        traj[sample_idx, -1, :, 0] + shift[-1],
        traj[sample_idx, -1, :, 1],
        color=colors[-1],
        alpha=0.7,
    )
    for i in range(traj.shape[-2]):
        x = traj[sample_idx, :, i, 0] + shift
        points = jnp.array([x, traj[sample_idx, :, i, 1]]).T.reshape(-1, 1, 2)
        segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors,
        norm = plt.Normalize(t.min(), t.max())
        lc = LineCollection(segments, cmap=cmap_name, norm=norm, alpha=0.4)
        # Set the values used for colormapping,
        lc.set_array(t),
        lc.set_linewidth(0.7),
        line = (ax.add_collection(lc),)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    ax.plot(traj[sample_idx, 0, :, 0], traj[sample_idx, 0, :, 1], color=colors[0])
    return ax


def add_start_to_end(xy, val_ax=-2):
    val = jnp.take(a=xy, axis=val_ax, indices=0)
    val = jnp.expand_dims(val, val_ax)
    return jnp.append(arr=xy, values=val, axis=val_ax)


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
