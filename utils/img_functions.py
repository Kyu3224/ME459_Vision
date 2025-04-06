import cv2
import numpy as np

def get_points_from_image(images, num_points, window_name, homogeneous = True):
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
    if not homogeneous:
        points = [points[i][:2] for i in range(len(points))]
    return np.array(points)