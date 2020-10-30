# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2
import math
from edgeDetection.util import draw_line

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def initializeHough(h, w, delta_d, delta_theta):
    min_d = w
    max_d = math.sqrt(math.pow(h,2) + math.pow(w,2))
    x = math.floor(180 / delta_theta)
    y = math.floor((min_d + max_d) / delta_d)


    print(h, w, min_d, max_d, x, y)

    return np.zeros([y, x], dtype=int)

def getEdges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    wide = cv2.Canny(blurred, 10, 200)
    # tight = cv2.Canny(blurred, 225, 250)
    # auto = auto_canny(blurred)
    return wide

def getParameterSpace(edges):
    delta_theta = 1
    delta_d = 1
    h, w = edges.shape

    H = initializeHough(h, w, delta_d, delta_theta)

    for (y, x) in np.argwhere(edges != 0):
        for theta in range(0, 180, delta_theta):
            theta_rad = np.deg2rad(theta)
            d = int(x * math.cos(theta_rad) + y * math.sin(theta_rad))
            H[d+w, theta] += 1

    return H/H.max()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread("images/dice.png")
    edges = getEdges(img);
    h, w = edges.shape

    # show the image
    cv2.imshow("Original Images", img)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)

    H = getParameterSpace(edges)

    cv2.imshow("Parameter Space", H)
    cv2.waitKey(0)

    # Draw Lines
    threshold = 0.4
    threshold_d = 20
    threshold_theta = np.deg2rad(7)

    num_lines = 80  # number of lines to draw
    idx_d_arr, idx_theta_arr = np.where(H > threshold)
    temp_idx = np.argsort(-1 * H[idx_d_arr, idx_theta_arr])
    lines = np.array([idx_d_arr[temp_idx], idx_theta_arr[temp_idx]]).T

    valueStore = []
    valueStore_d = []
    valueStore_theta = []

    for idx_d, idx_theta in lines[:num_lines]:
        d = idx_d - w
        theta = np.deg2rad(idx_theta)

        if np.abs(theta) < threshold_theta:
            theta = theta + 180
            d = d * -1

        for d_store, theta_store in valueStore:
            if (np.abs(d_store - d) < threshold_d) and (np.abs(theta_store - theta) < threshold_theta):
                #draw_line(img, d, theta, color=(0, 255, 255))
                break;
        else:
            valueStore.append([d, theta])
            draw_line(img, d, theta)

        print(d, np.rad2deg(theta))
        print()

    cv2.imshow("detected lines", img)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.waitKey(0)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
