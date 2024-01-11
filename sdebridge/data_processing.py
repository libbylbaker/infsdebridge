import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial


def order_points(unordered_points):
    points_for_waste = unordered_points.copy()
    p0 = points_for_waste[0, :].copy()
    points = [p0]
    for _ in range(len(unordered_points)):
        Midx = scipy.spatial.distance_matrix(p0[None, :], points_for_waste)
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
