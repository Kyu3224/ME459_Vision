import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import get_points_from_image

def cross_product(p1, p2):
    return np.cross(p1, p2)


def normalize_homogeneous(p):
    return p / p[2] if p[2] != 0 else p


def compute_vanishing_points(point):
    _v1 = normalize_homogeneous(cross_product(cross_product(point[0], point[1]), cross_product(point[2], point[3])))
    _v2 = normalize_homogeneous(cross_product(cross_product(point[0], point[2]), cross_product(point[1], point[3])))
    return _v1, _v2


def compute_height(ref_top, ref_bottom, human_bottom, human_top, R, v1, v2):
    vanishing_line = cross_product(v1, v2)
    v = normalize_homogeneous(cross_product(cross_product(ref_bottom, human_bottom), vanishing_line))
    t = normalize_homogeneous(cross_product(cross_product(v, human_top), cross_product(ref_top, ref_bottom)))
    cr = (t[1] - ref_bottom[1]) / (ref_top[1] - ref_bottom[1])
    return R * cr

def get_unique_filename(base_name, ext="png"):
    filename = f"{base_name}.{ext}"
    count = 1
    while os.path.exists(filename):
        filename = f"{base_name}_{count}.{ext}"
        count += 1
    return filename


if __name__ == "__main__":
    image = cv2.imread(f"{os.getcwd()}/hw2/hutme_ref.png")

    results_dir = f"{os.getcwd()}/hw2/results/"
    fig_name_1 = get_unique_filename(base_name=results_dir +"vanishing_pts")
    fig_name_2 = get_unique_filename(base_name=results_dir+"height_result")
    os.makedirs(results_dir, exist_ok=True)

    print("Select 4 points for vanishing point computation (p1 to p4)")

    num_sampling = 1
    points = []
    vanishing_points = list(get_points_from_image(image, 4 * num_sampling, "Select Vanishing Points"))
    for i in range(4):
        points.append(np.mean(vanishing_points[num_sampling*i:num_sampling*(i+1)], axis=0))

    # Compute vanishing points
    v1, v2 = compute_vanishing_points(points)

    # Show selected points and vanishing points
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter([points[0][0], points[1][0], points[2][0], points[3][0], v1[0], v2[0]],
                [points[0][1], points[1][1], points[2][1], points[3][1], v1[1], v2[1]],
                color=['red', 'red', 'red', 'red', 'blue', 'blue'],
                s=[50, 50, 50, 50, 50, 50])
    # red lines
    red_lines = [(points[0], points[1]), (points[2], points[3])]
    for start, end in red_lines:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2)

    # blue lines
    blue_lines = [(points[1], v1), (points[3], v1), (points[0], v2), (points[1], v2)]
    for start, end in blue_lines:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1)

    plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'g-',linewidth=1)

    plt.title("Selected Points and Vanishing Points")
    plt.savefig(fig_name_1, dpi=300, bbox_inches='tight')
    plt.show()

    print("Select reference and target points (ref_top, ref_bottom, human_bottom, human_top)")
    ref_points = get_points_from_image(image, 4, "Select Reference Points")
    ref_top, ref_bottom, human_bottom, human_top = ref_points

    R = 201  # Reference height in cm

    # Compute height
    H = compute_height(ref_top, ref_bottom, human_bottom, human_top, R, v1, v2)

    # Display result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter([points[0][0], points[1][0], points[2][0], points[3][0], ref_top[0], ref_bottom[0], human_bottom[0], human_top[0]],
                [points[0][1], points[1][1], points[2][1], points[3][1], ref_top[1], ref_bottom[1], human_bottom[1], human_top[1]], color='red')
    plt.title(f"Estimated Height: {H:.2f} cm")
    plt.savefig(fig_name_2, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Estimated Height: {H:.2f} cm")