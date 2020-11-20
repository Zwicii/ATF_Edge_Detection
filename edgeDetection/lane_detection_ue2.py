import cv2
import numpy as np
from edgeDetection.line_detection_ue1 import DELTA_D, DELTA_THETA
from edgeDetection.util import draw_line
from edgeDetection.line_detection_ue1 import getEdges
from PIL import Image, ImageDraw


THRESHOLD = 40
THRESHOLD_THETA = 25
THRESHOLD_D = 100

def getValueColorPosition(d, theta, w, h):
    # Geradengleichung aufstellen
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * d
    y0 = b * d

    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * a)
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * a)

    k = (y2-y1) / (x2-x1)
    d = y1 - k * x1 # y = k+x + d

    # calculate intersection with x-axis
    # h = k*x +d --> get x
    x_intersection = (h -d) / k
    pos = None
    val = None
    col = (160, 160, 160)

    if 0 <= x_intersection <= w:
        if x_intersection < w/2:
            col = (0, 235, 0) # green
            pos = (int(w/2 - 120), int(h - 30))
        elif x_intersection > w/2:
            col = (0, 0, 204) # red
            pos = (int(w / 2 + 90), int(h - 30))

        val = int(x_intersection - w/2) # distance to center
        return val, col, pos

    return None, col, None



if __name__ == '__main__':
    # cap = cv2.VideoCapture("data/highway1.mp4")
    # # cap = cv2.VideoCapture(0) # for webcam
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # TODO: do something with the frame
    #     cv2.imshow("Result", frame)
    #     # wait for 30ms -> results in ~30FPS
    #     cv2.waitKey(30)
    #
    # cv2.destroyAllWindows()
    # cap.release()

    #read image
    img = cv2.imread("data/highway1-3.png")
    ksize = (7, 7)
    img = cv2.blur(img, ksize)

    edges = getEdges(img)
    h, w = edges.shape
    cv2.imshow("edges", edges)
    cv2.waitKey(0)

    # mask
    mask = cv2.imread("data/highway1-2.png", cv2.IMREAD_GRAYSCALE)
    edges[mask == 0] = 0

    lines = cv2.HoughLines(edges, 1, np.pi / 180, THRESHOLD)

    valueStore = []

    for line in lines:
        for d, theta_rad in line:
            theta = np.rad2deg(theta_rad)

            originalValues = None

            if theta < THRESHOLD_THETA:
                originalValues = [d, theta]
                theta = theta + 180
                d = -d

            for d_store, theta_store in valueStore:
                if (np.abs(d_store - d) < THRESHOLD_D) and (np.abs(theta_store - theta) < THRESHOLD_THETA):
                    #draw_line(img, d, np.deg2rad(theta), color=(0, 255, 255))
                    break

                if originalValues is not None:
                    if (np.abs(originalValues[0] - d) < THRESHOLD_D) and (
                            np.abs(originalValues[1] - theta) < THRESHOLD_THETA):
                        #draw_line(img, d, np.deg2rad(theta), color=(0, 255, 255))
                        break

            else:
                if originalValues is not None:
                    valueStore.append([originalValues[0], originalValues[1]])

                valueStore.append([d, theta])
                # draw value and lines
                val, col, pos = getValueColorPosition(d, np.deg2rad(theta), w, h)
                draw_line(img, d, np.deg2rad(theta), color=col)
                if pos is not None:
                    cv2.putText(
                        img,  # numpy array on which text is written
                        str(val),  # text
                        pos,  # position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX,  # font family
                        0.8,  # font size
                        (0, 0, 0),  # font color
                        2)  # font stroke

    cv2.imshow("detected lines", img)
    cv2.waitKey(0)
