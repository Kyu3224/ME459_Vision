import cv2
import numpy as np


def get_points_from_image(images, num_points, window_name, homogeneous=True):
    points = []
    clone = images.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y, 1])
            print(f"Point selected: {x}, {y}")

            # Draw a circle and the index number on the image
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(clone, str(len(points)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 1, cv2.LINE_AA)

            # Show updated image
            cv2.imshow(window_name, clone)

            if len(points) == num_points:
                cv2.setMouseCallback(window_name, lambda *args: None)
                cv2.waitKey(500)  # slight pause before closing
                cv2.destroyWindow(window_name)

    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback)

    while len(points) < num_points:
        cv2.waitKey(1)

    if not homogeneous:
        points = [p[:2] for p in points]
    return np.array(points)
