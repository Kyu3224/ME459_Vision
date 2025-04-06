import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from utils import *
from numpy.linalg import svd, inv, qr

# Load image
img = cv2.imread(f'{os.getcwd()}/hw3/hw3_pattern.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show image and get 10 clicked points
plt.imshow(img_rgb)
pts = list(get_points_from_image(img_rgb, 10, "Click 10 points", homogeneous=False))
plt.close()

# Prepare image coordinates (homogeneous)
x = np.ones((3, 10))
for i, (xi, yi) in enumerate(pts):
    x[0, i] = xi
    x[1, i] = yi

# Define corresponding 3D points (homogeneous)
X = np.array([
    [0, 0, 0, 1],
    [12, 0, 0, 1],
    [0, 12, 0, 1],
    [0, 0, 12, 1],
    [24, 0, 24, 1],
    [0, 24, 24, 1],
    [24, 24, 0, 1],
    [48, 0, 48, 1],
    [0, 48, 48, 1],
    [48, 48, 0, 1]
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
U, S, Vt = svd(A)
P = Vt[-1, :].reshape(3, 4)

# Compute camera center C
U, S, Vt = svd(P)
C = Vt[-1, :]
C = C[:3] / C[3]

# Decompose projection matrix
M = P[:, :3]
R_inv, K_inv = qr(inv(M))  # QR of inv(M)

# Normalize K
K = inv(K_inv)
K = K / K[2, 2]

# Compute R
R = inv(R_inv)

# Translation vector
t = -R @ C

# Print results
print("Projection matrix P:\n", P)
print("Camera center C:\n", C)
print("Intrinsic matrix K:\n", K)
print("Rotation matrix R:\n", R)
print("Translation vector t:\n", t)
