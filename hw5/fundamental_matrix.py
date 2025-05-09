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


def normalize_points(x):
    """
    Normalize 2D homogeneous points so that centroid is at origin
    and average distance from origin is sqrt(2).

    Args:
        x: (3, N) homogeneous coordinates

    Returns:
        x_norm: (3, N) normalized homogeneous coordinates
        T: (3, 3) normalization transform
    """
    x = x / x[2]
    centroid = np.mean(x[:2], axis=1)
    centered = x[:2] - centroid[:, np.newaxis]
    avg_dist = np.mean(np.linalg.norm(centered, axis=0))
    scale = np.sqrt(2) / avg_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    x_norm = T @ x
    return x_norm, T


def compute_fundamental_matrix_normalized(x1, x2):
    """
    Estimate the Fundamental matrix F using the normalized 8-point algorithm.

    Args:
        x1: (3, N) homogeneous coordinates in image 1
        x2: (3, N) homogeneous coordinates in image 2

    Returns:
        F: (3, 3) fundamental matrix
    """
    assert x1.shape[1] == x2.shape[1] >= 8, "At least 8 point correspondences are required"

    # Normalize points
    x1_norm, T1 = normalize_points(x1)
    x2_norm, T2 = normalize_points(x2)

    N = x1.shape[1]
    A = np.zeros((N, 9))

    for i in range(N):
        x1x, x1y = x1_norm[0, i], x1_norm[1, i]
        x2x, x2y = x2_norm[0, i], x2_norm[1, i]
        A[i] = [
            x1x * x2x, x1y * x2x, x2x,
            x1x * x2y, x1y * x2y, x2y,
            x1x, x1y, 1
        ]

    # Solve Af = 0 using SVD
    U, D, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U_f, D_f, Vt_f = np.linalg.svd(F_hat)
    D_f[2] = 0
    F_hat_rank2 = U_f @ np.diag(D_f) @ Vt_f

    # Denormalize
    F = T2.T @ F_hat_rank2 @ T1

    # Normalize (optional: F[2,2] not guaranteed ≠ 0 after normalization)
    return F / np.linalg.norm(F)


def draw_epipolar_lines(img, F, ref_pts, pts, method, img_name, which_img='left'):
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
    save_dir = os.path.join(os.getcwd(), "hw5", "results", method, img_name)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir+".png")
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


def extract_feature_matches(img1, img2, method='SIFT', max_matches=500):
    # 1. Feature Detector 선택
    if method == 'SIFT':
        detector = cv2.SIFT_create()
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_matches)
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 2. 특징점 및 디스크립터 추출
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # 3. Matcher 선택 (특성에 따라 다름)
    if method in ['SIFT', 'AKAZE']:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 4. 매칭 수행 및 정렬
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

    # 5. 대응점 좌표 추출
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])  # (N, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])  # (N, 2)

    # 6. Homogeneous coordinates로 변환
    pts1_h = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))  # (3, N)
    pts2_h = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))  # (3, N)

    return pts1_h, pts2_h, matches, kp1, kp2


img_name = "c"
mode = "ORB" # Available options: SIFT, AKAZE, ORB, Manual
num_pts = 15 # Should be equal or bigger than 8.

assert num_pts >= 8

img1 = cv2.cvtColor(cv2.imread(f'{os.getcwd()}/hw5/hw5_img/{img_name}1.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(f'{os.getcwd()}/hw5/hw5_img/{img_name}2.jpg'), cv2.COLOR_BGR2RGB)

if mode == "Manual":
    points = collect_points(img_name, [1,2], num_pts)
    print(points)

    x1_pts = np.array(points[f"{img_name}1"]).T  # (3, N)
    x2_pts = np.array(points[f"{img_name}2"]).T  # (3, N)
else:
    x1_pts, x2_pts, matches, kp1, kp2 = extract_feature_matches(img1, img2, method=mode, max_matches=num_pts)

F_basic = compute_fundamental_matrix(x1_pts, x2_pts)
F_norm = compute_fundamental_matrix_normalized(x1_pts, x2_pts)

# Convert to (N, 2) float
pts1 = (x1_pts[:2] / x1_pts[2]).T
pts2 = (x2_pts[:2] / x2_pts[2]).T

# RANSAC-based estimation -> To improve performance
F_ransac, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0)

# Select inliers
x1_in = x1_pts[:, inlier_mask.ravel() == 1]
x2_in = x2_pts[:, inlier_mask.ravel() == 1]

# Compute refined normalized fundamental matrix
F_refined = compute_fundamental_matrix_normalized(x1_in, x2_in)

# Define fundamental matrices and associated point correspondences
epipolar_configs = [
    ("basic", "Fundamental Matrix F", F_basic, x1_pts, x2_pts),
    ("norm", "Fundamental Matrix F with normalization", F_norm, x1_pts, x2_pts),
    ("ransac", "Fundamental Matrix F with RANSAC", F_ransac, x1_in, x2_in),
    ("refine", "Fundamental Matrix F with RANSAC + normalization", F_refined, x1_in, x2_in),
]

# Loop through each configuration
for label, description, F, pts1, pts2 in epipolar_configs:
    print(f"{description}:\n{F}\n")

    draw_epipolar_lines(img1, F, pts1, pts2, method=mode, img_name=f"{img_name}1/{label}", which_img='left')
    draw_epipolar_lines(img2, F, pts2, pts1, method=mode, img_name=f"{img_name}2/{label}", which_img='right')

    l1 = F @ pts2
    l2 = F.T @ pts1

    dist1 = compute_point_line_distances(pts1, l1)
    dist2 = compute_point_line_distances(pts2, l2)
    mean_residual, residuals = compute_epipolar_residuals(F, pts1, pts2)

    print(f"Distances from x1 to its epipolar lines({label}):\n", dist1)
    print(f"Mean Distance for image 1({label}):\n", dist1.mean())
    print(f"Distances from x2 to its epipolar lines({label}):\n", dist2)
    print(f"Mean Distance for image 2({label}):\n", dist2.mean())
    print(f"Mean residual error({label}):\n {mean_residual:.6f}")
    print(f"Individual residuals({label}):\n {residuals}")
