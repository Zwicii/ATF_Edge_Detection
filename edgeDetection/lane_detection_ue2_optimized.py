import cv2
import numpy as np

from edgeDetection.line_detection_ue1 import DELTA_D, DELTA_THETA
from edgeDetection.util import draw_line
from edgeDetection.line_detection_ue1 import getEdges, initializeHough


THRESHOLD = 40
THRESHOLD_THETA = 30
THRESHOLD_D = 100
THRESHOLD_IMAGE_W = 50


def getParameterSpace(edges):

    h, w = edges.shape

    edges_matrix = np.array(np.argwhere(edges != 0))

    x_matrix = np.tile(edges_matrix[:, 1], ((180 // DELTA_THETA), 1)).T
    y_matrix = np.tile(edges_matrix[:, 0], ((180 // DELTA_THETA), 1)).T

    theta = np.array(range(0, 180, DELTA_THETA))
    theta_rad = np.deg2rad(theta)
    theta_matrix = np.tile(theta_rad, (len(x_matrix), 1))

    d_matrix = x_matrix * np.cos(theta_matrix) + y_matrix * np.sin(theta_matrix)
    d_matrix = (d_matrix + w) // DELTA_D
    d_matrix = d_matrix.astype(int)

    d_matrix = np.transpose(d_matrix)
    N = d_matrix.max() + 1
    id = d_matrix + (N * np.arange(d_matrix.shape[0]))[:, None]
    H = np.bincount(id.ravel(), minlength=N * d_matrix.shape[0]).reshape(-1, N).T

    return H


def processFrame(img, mask):
    edges = getEdges(img)
    h, w = edges.shape

    # mask
    #mask = cv2.imread("data/mask.png", cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        edges[mask == 0] = 0

    H = getParameterSpace(edges)
    idx_d_arr, idx_theta_arr = np.where(H > THRESHOLD)
    temp_idx = np.argsort(-1 * H[idx_d_arr, idx_theta_arr])
    lines = np.array([idx_d_arr[temp_idx], idx_theta_arr[temp_idx]]).T

    valueStore = []
    pos_left = (round(w / 2 - 150), round(h - 10))
    pos_right = (round(w / 2 + 50), round(h - 10))
    left_lane = None
    right_lane = None

    for idx_d, idx_theta in lines:

        d = idx_d * DELTA_D - w
        theta = (idx_theta * DELTA_THETA)
        originalValues = None

        if theta < THRESHOLD_THETA:
            originalValues = [d, theta]
            theta = theta + 180
            d = -d

        for d_store, theta_store in valueStore:
            if (np.abs(d_store - d) < THRESHOLD_D) and (np.abs(theta_store - theta) < THRESHOLD_THETA):
                # draw_line(img, d, np.deg2rad(theta), color=(0, 255, 255))
                break

            if originalValues is not None:
                if (np.abs(originalValues[0] - d) < THRESHOLD_D) and (
                        np.abs(originalValues[1] - theta) < THRESHOLD_THETA):
                    # draw_line(img, d, np.deg2rad(theta), color=(0, 255, 255))
                    break

        else:
            if originalValues is not None:
                valueStore.append([originalValues[0], originalValues[1]])

            valueStore.append([d, theta])
            # draw value and lines
            val, type = getDrawingParameters(d, np.deg2rad(theta), w, h)

            if type == "left_lane":
                if left_lane is None or left_lane[0] < val:
                    left_lane = val, d, theta
            elif type == "right_lane":
                if right_lane is None or right_lane[0] > val:
                    right_lane = val, d, theta
            else:
                draw_line(img, d, np.deg2rad(theta), color=(160, 160, 160))

    if left_lane is not None:
        draw_line(img, left_lane[1], np.deg2rad(left_lane[2]), color=(0, 235, 0))
        cv2.putText(
            img,  # numpy array on which text is written
            "L: " + str(left_lane[0])  + " px",  # text
            pos_left,  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.5,  # font size
            (0, 0, 0),  # font color
            1)  # font stroke

    if right_lane is not None:
        draw_line(img, right_lane[1], np.deg2rad(right_lane[2]), color=(0, 0, 204))
        cv2.putText(
            img,  # numpy array on which text is written
            "R: " + str(right_lane[0]) + " px",  # text
            pos_right,  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.5,  # font size
            (0, 0, 0),  # font color
            1)  # font stroke
    return img, left_lane[0] if left_lane is not None else None, right_lane[0] if right_lane is not None else None

def getDrawingParameters(d, theta, w, h):
    x_intersection = (d - h * np.sin(theta)) / np.cos(theta)
    type = None
    val = None

    if -THRESHOLD_IMAGE_W <= x_intersection <= w + THRESHOLD_IMAGE_W:
        if x_intersection < w/2 and 20 < np.rad2deg(theta) < 60: # left lane
            type = "left_lane"
        elif x_intersection > w/2 and 100 < np.rad2deg(theta) < 160: # right lane
            type = "right_lane"

        val = round(x_intersection - w/2) # distance to center

    return val, type

if __name__ == '__main__':
    cap = cv2.VideoCapture("data/highway1.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ksize = (3, 3)
        frame = cv2.blur(frame, ksize)
        processedFrame = processFrame(frame)
        cv2.imshow("Result", processedFrame)
        # wait for 30ms -> results in ~30FPS
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    cap.release()


