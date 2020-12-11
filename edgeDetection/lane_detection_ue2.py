import cv2
import numpy as np
from edgeDetection.line_detection_ue1 import DELTA_D, DELTA_THETA
from edgeDetection.util import draw_line
from edgeDetection.line_detection_ue1 import getEdges


THRESHOLD = 40
THRESHOLD_THETA = 35
THRESHOLD_D = 120
THRESHOLD_IMAGE_W = 50



def processFrame(img, mask):
    edges = getEdges(img)
    h, w = edges.shape
    #cv2.imshow("edges", edges)

    # mask
    if mask is not None:
        #mask = cv2.imread("data/mask.png", cv2.IMREAD_GRAYSCALE)
        edges[mask == 0] = 0

    lines = cv2.HoughLines(edges, 1, np.pi / 180, THRESHOLD)

    valueStore = []
    #old configs for ex02
    #pos_left = (round(w / 2 - 120), round(h - 30))
    #pos_right = (round(w / 2 + 90), round(h - 30))

    #new configs for ex03
    pos_left = (round(w / 2 - 150), round(h - 10))
    pos_right = (round(w / 2 + 50), round(h - 10))
    left_lane = None
    right_lane = None

    if lines is not None:
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
                "L: " + str(left_lane[0]),  # text
                pos_left,  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                0.5,  # font size
                (0, 0, 0),  # font color
                1)  # font stroke

        if right_lane is not None:
            draw_line(img, right_lane[1], np.deg2rad(right_lane[2]), color=(0, 0, 204))
            cv2.putText(
                img,  # numpy array on which text is written
                "R: " + str(right_lane[0]),  # text
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
        if x_intersection < w/2 and 40 < np.rad2deg(theta) < 80: # left lane
            type = "left_lane"
        elif x_intersection > w/2 and 100 < np.rad2deg(theta) < 160: # right lane
            type = "right_lane"

        val = round(x_intersection - w/2) # distance to center

    return val, type



if __name__ == '__main__':
    #read image
    img = cv2.imread("data/highway1-3.png")
    ksize = (6, 6)
    blurImg = cv2.blur(img, ksize)
    processedImage = processFrame(blurImg)
    cv2.imshow("detected filtered lines", processedImage)
    cv2.waitKey(0)

    #video
    cap = cv2.VideoCapture("data/highway1.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ksize = (4, 4)
        frame = cv2.blur(frame, ksize)
        processedFrame = processFrame(frame)
        cv2.imshow("Result", processedFrame)
        # wait for 30ms -> results in ~30FPS
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    cap.release()

