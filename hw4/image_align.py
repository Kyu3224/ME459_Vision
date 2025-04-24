import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import *

def collect_points(img_name, image_seq, num_pts):
    pts = {}
    for idx in image_seq:
        _name = img_name + str(idx)
        img = cv2.imread(f'{os.getcwd()}/hw4/hw4_img/{_name}.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        _pts = list(get_points_from_image(img_rgb, num_pts,
                                          f"Click 4 points : {_name}"))
        plt.close()
        if _name not in pts:
            pts[_name] = _pts
        else:
            pts[_name+'new'] = _pts
    return pts

def get_homography(src_pts, dst_pts):
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def get_homography_set(image_name, point_set):
    homography_set = {}
    homography_set['12'] = get_homography(point_set[image_name+'1'], point_set[image_name+'2'])
    homography_set['32'] = get_homography(point_set[image_name+'3'], point_set[image_name+'2new'])
    return homography_set


def warp_images(images, homographies):
    # Base Image
    center_idx = 1
    base_name = list(images.keys())[center_idx]
    base_img = images[base_name]

    height, width = base_img.shape[:2]
    out_size = (width * 3, height * 3)
    offset = (width, height)

    panorama = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
    panorama[offset[1]:offset[1] + height, offset[0]:offset[0] + width] = base_img

    # Left Image
    left_name = list(images.keys())[center_idx - 1]
    H_left = np.array([[1, 0, offset[0]],
                       [0, 1, offset[1]],
                       [0, 0, 1]]) @ homographies['12']
    # https://docs.opencv.org/4.x/da/d54/group__imgproc__
    # transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121ac97d8e4880d8b5d509e96825c7522deb
    warped_left = cv2.warpPerspective(images[left_name], H_left, out_size)
    mask_left = (warped_left > 0).astype(np.uint8)
    mask_left = np.any(mask_left, axis=2, keepdims=True)
    panorama = np.where(mask_left, warped_left, panorama)

    # Right Image
    right_name = list(images.keys())[center_idx + 1]
    H_right = np.array([[1, 0, offset[0]],
                        [0, 1, offset[1]],
                        [0, 0, 1]]) @ homographies['32']
    warped_right = cv2.warpPerspective(images[right_name], H_right, out_size)
    mask_right = (warped_right > 0).astype(np.uint8)
    mask_right = np.any(mask_right, axis=2, keepdims=True)
    panorama = np.where(mask_right, warped_right, panorama)

    return panorama

def load_images(img_name, num_images):
    imgs = {}
    for i in range(1, num_images + 1):
        name = f'{img_name}{i}'
        img = cv2.imread(f'{os.getcwd()}/hw4/hw4_img/{name}.jpg')
        imgs[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return imgs

img_name = "c"
num_pts = 4

points = collect_points(img_name, [1,2,3,2], num_pts)

homographies = get_homography_set(img_name, points)

images = load_images(img_name, 3)
stitched_image = warp_images(images, homographies)

results_dir = f"{os.getcwd()}/hw4/results/"
fig_name = get_unique_filename(base_name=results_dir + img_name + f"_{num_pts}pts")

# 결과 출력
plt.figure(figsize=(12, 6))
plt.imshow(stitched_image)
plt.axis('off')
plt.title('Stitched Panorama')
plt.savefig(fig_name, dpi=300, bbox_inches='tight')
plt.show()

