import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from utils import *
from numpy.linalg import svd, inv, qr
from scipy.linalg import rq

use_rq = True

# Load image
img = cv2.imread(f'{os.getcwd()}/hw3/hw3_pattern.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show image and get 10 clicked points
plt.imshow(img_rgb)
pts = list(get_points_from_image(img_rgb, 10, "Click 10 points", homogeneous=True))
plt.close()

c_true = np.array([166.20, 141.46, 170.08])

x = np.array(pts).T

# Define corresponding 3D points (homogeneous)
X = np.array([
    [0, 0, 0, 1],
    [12, 0, 0, 1],
    [0, 12, 0, 1],
    [0, 0, 12, 1],
    [24, 24, 0, 1],
    [0, 24, 24, 1],
    [24, 0, 24, 1],
    [48, 48, 0, 1],
    [0, 48, 48, 1],
    [48, 0, 48, 1]
])

# Construct matrix A
A = np.zeros((20, 12))
for i in range(10):
    X_i = X[i, :]
    xi = x[0, i]
    yi = x[1, i]
    A[2 * i, 0:4] = X_i
    A[2 * i, 8:12] = -xi * X_i
    A[2 * i + 1, 4:8] = X_i
    A[2 * i + 1, 8:12] = -yi * X_i

# Estimate projection matrix P
_, _, Vt = svd(A)
P = Vt[-1, :].reshape(3, 4)

# Compute camera center C
_, _, Vt = svd(P)
C = Vt[-1, :]
C = C[:3] / C[3]

# Decompose projection matrix
M = P[:, :3]
if use_rq:
    K, R = rq(M)
else:
    R_inv, K_inv = qr(inv(M))  # QR of inv(M)
    K = inv(K_inv)
    R = inv(R_inv)

# Normalize K
K = K / K[2, 2]

# Translation vector
t = -R @ C

abs_error = c_true - C
relative_error = np.linalg.norm(abs_error) / np.linalg.norm(c_true) * 100

# Print results
print("Projection matrix P:\n", P)
print("==============================================================================")
print(f"True Camera center:{c_true}")
print(f"Estimated Camera center C: {C}")
print(f"Absolute error: {abs_error}")
print(f"Relative error: {relative_error} %")
print("==============================================================================")
print("Intrinsic matrix K:\n", K)
print("Rotation matrix R:\n", R)
print("Translation vector t:\n", t)
