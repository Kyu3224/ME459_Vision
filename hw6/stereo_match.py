import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

def SAD(window1, window2):
    return np.sum(np.abs(window1 - window2))


def SSD(window1, window2):
    return np.sum((window1 - window2) ** 2)


def NCC(window1, window2):
    mean1 = np.mean(window1)
    mean2 = np.mean(window2)
    numerator = np.sum((window1 - mean1) * (window2 - mean2))
    denominator = np.std(window1) * np.std(window2) * window1.size
    return -numerator / denominator if denominator != 0 else 0


def compute_disparity_row(args):
    i, img1, img2, gap, max_disp, metric = args
    height, width = img1.shape
    row_disp = np.zeros(width)

    for j in range(gap, width - gap):
        min_cost = float('inf')
        best_disp = 0
        window1 = img1[i - gap:i + gap + 1, j - gap:j + gap + 1]

        for d in range(max(0, j - max_disp), j):
            col_start = d - gap
            col_end = d + gap + 1

            if col_start < 0 or col_end > width:
                continue

            window2 = img2[i - gap:i + gap + 1, col_start:col_end]

            if window2.shape != window1.shape:
                continue

            if metric == 'SAD':
                cost = SAD(window1, window2)
            elif metric == 'SSD':
                cost = SSD(window1, window2)
            elif metric == 'NCC':
                cost = NCC(window1, window2)
            else:
                raise ValueError("Unknown metric")

            if cost < min_cost:
                min_cost = cost
                best_disp = j - d

        row_disp[j] = best_disp
    return i, row_disp

def compute_disparity_map(img1, img2, gap=2, max_disp=100, metric='SAD'):
    height, width = img1.shape
    disparity_map = np.zeros((height, width))

    args_list = [(i, img1, img2, gap, max_disp, metric) for i in range(gap, height - gap)]

    with Pool(processes=cpu_count()) as pool:
        for i, row_disp in pool.imap_unordered(compute_disparity_row, args_list):
            disparity_map[i, :] = row_disp
            print(f"Scanline {i}/{height - gap - 1}")

    return disparity_map


if __name__ == '__main__':
    img_name = 'A'

    # Main execution
    img1 = cv2.imread(f'{os.getcwd()}/hw6/hw6_img/set_{img_name}/left_{img_name}.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'{os.getcwd()}/hw6/hw6_img/set_{img_name}/right_{img_name}.png', cv2.IMREAD_GRAYSCALE)
    if img_name == 'A':
        img3 = cv2.imread(f'{os.getcwd()}/hw6/hw6_img/set_A/disp_A.png',
                          cv2.IMREAD_GRAYSCALE).astype(np.float32) / 4.0
    else:
        img3 = None

    height, width = img1.shape
    gap = 2
    metric = 'NCC' # Available options are SAD, SSD, NCC

    disparity_map = compute_disparity_map(img1, img2, gap=gap, metric=metric)

    # Visualization
    plt.figure(figsize=(10, 4))
    if img_name == 'A':
        plt.subplot(1, 2, 1)
    plt.title("Computed Disparity")
    plt.imshow(disparity_map, cmap='gray', vmin=0, vmax=63.75)
    plt.colorbar()

    plt.axis('off')
    save_dir = os.path.join(os.getcwd(), "hw6", "results", metric, img_name)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir + ".png")

    if img_name == 'A':
        plt.subplot(1, 2, 2)
        plt.title("Ground Truth Disparity")
        plt.imshow(img3, cmap='gray', vmin=0, vmax=63.75)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        error = disparity_map - img3
        rmse = np.sqrt(np.mean(error ** 2))
        print(f"RMSE: {rmse:.4f}")

        # 3-pixel error rate
        correct = np.abs(error) < 3
        three_pixel_error_rate = np.sum(correct) / (height * width) * 100
        print(f"3-pixel Error Rate: {three_pixel_error_rate:.2f}%")

        # Error map visualization
        plt.figure()
        plt.title("Error Map")
        plt.imshow(error, cmap='jet')
        plt.colorbar()
        plt.show()
