import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


def sample_ellipse(
    num_points: int,
    scale: float = 1.0,
    shifts: np.ndarray = np.array([0.0, 0.0]),
    a: float = 1.0,
    b: float = 1.0,
) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return scale * np.stack([x, y], axis=0) + shifts[:, None]
    # return (scale * np.stack([x, y], axis=1) + shifts).flatten()


def order_points(unordered_points):
    points_for_waste = unordered_points.copy()
    p0 = points_for_waste[0, :].copy()
    points = [p0]
    for _ in range(len(unordered_points)):
        Midx = spatial.distance_matrix(p0[None, :], points_for_waste)
        Midx[Midx == 0.0] = 100
        idx = np.argmin(Midx)
        p0 = points_for_waste[idx, :].copy()
        points_for_waste[idx, :] = 100000
        points.append(p0)
    return np.asarray(points)


def get_points(im_path: str):
    image = cv2.imread(im_path)
    gray_scale_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_in_im = cv2.Canny(gray_scale_im, 50, 200)
    pixels = np.argwhere(edges_in_im == 255)
    points = np.array([1, -1]) * pixels[:, ::-1]  # Rotate points
    return points


def _interpolate(path, remove_pts):
    ordered = np.asarray(order_points(get_points(path)), dtype=float)[:remove_pts]
    ordered[:, 0] = _scale(ordered[:, 0])
    ordered[:, 1] = _scale(ordered[:, 1])
    x1 = np.interp(
        np.arange(0.0, len(ordered), 0.05), np.arange(len(ordered)), ordered[:, 0]
    )
    x2 = np.interp(
        np.arange(0.0, len(ordered), 0.05), np.arange(len(ordered)), ordered[:, 1]
    )
    return x1, x2


def _scale(points):
    maxpt = np.max(points)
    minpt = np.min(points)
    return (points - minpt) / (maxpt - minpt)


def butterfly1_pts():
    path_b1 = "../data/inverted_butterfly1.png"
    remove_pts_b1 = -466
    return _interpolate(path_b1, remove_pts_b1)


def butterfly2_pts():
    path_b2 = "../data/inverted_butterfly2.png"
    remove_pts_b2 = -70
    return _interpolate(path_b2, remove_pts_b2)


def butterfly_bw_pts():
    path_b2 = "../data/inverted_butterfly_bw.png"
    remove_pts_b2 = -1
    return _interpolate(path_b2, remove_pts_b2)


def butterfly_honrathi_pts():
    path_b2 = "../data/inverted_butterfly_honrathi.jpg"
    remove_pts_b2 = -38
    return _interpolate(path_b2, remove_pts_b2)


def butterfly_amasina_pts():
    path_b2 = "../data/inverted_butterfly_amasina.png"
    remove_pts_b2 = -73
    return _interpolate(path_b2, remove_pts_b2)
