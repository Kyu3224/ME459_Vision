import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import *

def collect_points(name, image_seq, num_pts):
    pts = {}
    for idx in image_seq:
        _name = name + str(idx)
        img = cv2.imread(f'{os.getcwd()}/hw5/hw5_img/{_name}.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        _pts = list(get_points_from_image(img_rgb, num_pts,f"Click {num_pts} points : {_name}"))
        plt.close()
        if _name not in pts:
            pts[_name] = _pts
        else:
            pts[_name+'new'] = _pts
    return pts

def compute_fundamental_matrix(x1, x2):
    """
    Estimate the Fundamental matrix F using the 8-point algorithm.

    Args:
        x1: (3, N) homogeneous coordinates in image 1
        x2: (3, N) homogeneous coordinates in image 2

    Returns:
        F: (3, 3) fundamental matrix
    """
    assert x1.shape[1] == x2.shape[1], "Number of points must match"
    N = x1.shape[1]
    A = np.zeros((N, 9))

    for i in range(N):
        x1x, x1y = x1[0, i], x1[1, i]
        x2x, x2y = x2[0, i], x2[1, i]
        A[i] = [
            x1x * x2x, x1y * x2x, x2x,
            x1x * x2y, x1y * x2y, x2y,
            x1x,       x1y,       1
        ]

    # Solve Af = 0 using SVD
    U, D, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    U_f, D_f, Vt_f = np.linalg.svd(F)
    D_f[2] = 0  # Set smallest singular value to zero
    F_rank2 = U_f @ np.diag(D_f) @ Vt_f

    return F_rank2 / F_rank2[2, 2]  # Normalize for stability


def draw_epipolar_lines(img, F, ref_pts, pts, img_name, which_img='left'):
    """
    Draw epipolar lines corresponding to pts on the image.

    Args:
        img: (H, W, 3) image on which to draw epipolar lines
        F: (3, 3) fundamental matrix
        pts: (3, N) homogeneous coordinates in the *other* image
        img_name: base name for saving image
        which_img: 'left' or 'right'
    """
    img_draw = img.copy()
    H, W, _ = img.shape

    if which_img == 'left':
        lines = F @ pts  # Epipolar lines in left image
    else:
        lines = F.T @ pts  # Epipolar lines in right image

    fig, ax = plt.subplots()
    ax.imshow(img_draw)

    for i in range(lines.shape[1]):
        a, b, c = lines[:, i]

        if abs(b) > 1e-8:
            # y = -(a*x + c)/b
            x_vals = np.array([0, W])
            y_vals = -(a * x_vals + c) / b
        else:
            # vertical line
            x_vals = np.full(2, -c / a)
            y_vals = np.array([0, H])

        # Only draw if points are in image range
        if np.all((0 <= y_vals) & (y_vals <= H)):
            ax.plot(x_vals, y_vals, 'g', linewidth=1)

        # Draw the original corresponding point
        ax.scatter(ref_pts[0, i], ref_pts[1, i], color='r', s=10)

    plt.title(f"Epipolar lines in {which_img} image")
    plt.axis('off')
    save_dir = os.path.join(os.getcwd(), "hw5", "results")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{img_name}_result.png"))
    plt.close()


def compute_epipolar_residuals(F, x1_h, x2_h):
    """
    Compute the average epipolar constraint residuals: |x2.T @ F @ x1|

    Parameters:
        F (np.ndarray): Fundamental matrix of shape (3, 3)
        x1_h (np.ndarray): Homogeneous coordinates in image 1, shape (3, N)
        x2_h (np.ndarray): Homogeneous coordinates in image 2, shape (3, N)

    Returns:
        float: Mean residual error over all correspondences
        np.ndarray: Residuals for each correspondence (N,)
    """
    assert F.shape == (3, 3), "F must be a 3x3 matrix"
    assert x1_h.shape[0] == 3 and x2_h.shape[0] == 3, "Points must be in homogeneous coordinates"

    # Compute the residuals: each is a scalar value |x2ᵢ.T @ F @ x1ᵢ|
    residuals = np.abs(np.einsum('ij,ji->i', x2_h.T, F @ x1_h))
    mean_residual = np.mean(residuals)
    return mean_residual, residuals

def compute_point_line_distances(pts, lines):
    """
    point-line distance: |l^T x| / sqrt(a^2 + b^2)

    Args:
        pts: (3, N)
        lines: (3, N) epipolar lines

    Returns:
        distances: (N,)
    """
    numerator = np.abs(np.sum(pts * lines, axis=0))  # (N,)
    denominator = np.sqrt(lines[0, :] ** 2 + lines[1, :] ** 2)
    return numerator / denominator


img_name = "c"
num_pts = 8 # Should be equal or bigger than 8.

assert num_pts >= 8

points = collect_points(img_name, [1,2], num_pts)
print(points)

x1_pts = np.array(points[f"{img_name}1"]).T  # (3, N)
x2_pts = np.array(points[f"{img_name}2"]).T  # (3, N)

F = compute_fundamental_matrix(x1_pts, x2_pts)

print("Fundamental Matrix F:\n", F)

img1 = cv2.cvtColor(cv2.imread(f'{os.getcwd()}/hw5/hw5_img/{img_name}1.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(f'{os.getcwd()}/hw5/hw5_img/{img_name}2.jpg'), cv2.COLOR_BGR2RGB)

draw_epipolar_lines(img1, F, x1_pts, x2_pts, img_name=img_name+"1", which_img='left')   # x2 → img1
draw_epipolar_lines(img2, F, x2_pts, x1_pts, img_name=img_name+"2", which_img='right')  # x1 → img2

l1 = F @ x2_pts
l2 = F.T @ x1_pts

dist1 = compute_point_line_distances(x1_pts, l1)
dist2 = compute_point_line_distances(x2_pts, l2)

print("Distances from x1 to its epipolar lines:\n", dist1)
print("Distances from x2 to its epipolar lines:\n", dist2)

mean_residual, residuals = compute_epipolar_residuals(F, x1_pts, x2_pts)

print(f"Mean residual error: {mean_residual:.6f}")
print(f"Individual residuals:\n{residuals}")
