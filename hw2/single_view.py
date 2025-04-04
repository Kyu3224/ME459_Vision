import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cross_product(p1, p2):
    return np.cross(p1, p2)


def normalize_homogeneous(p):
    return p / p[2] if p[2] != 0 else p


def compute_vanishing_points(p1, p2, p3, p4):
    v1 = normalize_homogeneous(cross_product(cross_product(p1, p2), cross_product(p3, p4)))
    v2 = normalize_homogeneous(cross_product(cross_product(p1, p3), cross_product(p2, p4)))
    return v1, v2


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


def get_points_from_image(images, num_points, window_name):
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y, 1])
            print(f"Point selected: {x}, {y}")
            if len(points) == num_points:
                cv2.setMouseCallback(window_name, lambda *args: None)
                cv2.destroyWindow(window_name)

    clone = images.copy()
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback)
    while len(points) < num_points:
        cv2.waitKey(1)
    return np.array(points)


if __name__ == "__main__":
    image = cv2.imread(f"{os.getcwd()}/hw2/hutme_ref.png")

    results_dir = f"{os.getcwd()}/hw2/results/"
    fig_name_1 = get_unique_filename(base_name=results_dir +"vanishing_pts")
    fig_name_2 = get_unique_filename(base_name=results_dir+"height_result")
    os.makedirs(results_dir, exist_ok=True)

    print("Select 4 points for vanishing point computation (p1 to p4)")
    vanishing_points = get_points_from_image(image, 4, "Select Vanishing Points")
    p1, p2, p3, p4 = vanishing_points

    # Compute vanishing points
    v1, v2 = compute_vanishing_points(p1, p2, p3, p4)

    # Show selected points and vanishing points
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter([p1[0], p2[0], p3[0], p4[0], v1[0], v2[0]],
                [p1[1], p2[1], p3[1], p4[1], v1[1], v2[1]],
                color=['red', 'red', 'red', 'red', 'blue', 'blue'],
                s=[50, 50, 50, 50, 50, 50])
    # red lines
    red_lines = [(p1, p2), (p3, p4)]
    for start, end in red_lines:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2)

    # blue lines
    blue_lines = [(p2, v1), (p4, v1), (p1, v2), (p2, v2)]
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
    plt.scatter([p1[0], p2[0], p3[0], p4[0], ref_top[0], ref_bottom[0], human_bottom[0], human_top[0]],
                [p1[1], p2[1], p3[1], p4[1], ref_top[1], ref_bottom[1], human_bottom[1], human_top[1]], color='red')
    plt.title(f"Estimated Height: {H:.2f} cm")
    plt.savefig(fig_name_2, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Estimated Height: {H:.2f} cm")