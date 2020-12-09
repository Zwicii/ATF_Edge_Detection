import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from flask import Flask
from flask_socketio import SocketIO
from edgeDetection.lane_detection_ue2 import processFrame
from edgeDetection.lane_detection_ue2_optimized import processFrame as our_process_frame

app = Flask(__name__)
socket_io = SocketIO(app)


throttle = 0
angle = 0
step = -0.05
COUNTER = 0

def updateAngelAndThrottle(speed):
    global throttle
    global angle
    global step

    if speed >= 15:
        throttle = 0
    elif speed >= 10:
        throttle = 0.5
    else:
        throttle = 1

    angle += step
    if angle >= 1:
        angle = 0
        step = -0.05
    elif angle <= -1:
        angle = 0
        step = 0.05



@socket_io.on('telemetry')
def telemetry(data):
    global COUNTER
    if data:
        speed = float(data["speed"])
        image = cv2.cvtColor(np.array(Image.open(BytesIO(base64.b64decode(data["image"])))), cv2.COLOR_RGB2BGR)

        dist_left, dist_right = lane_detection_opencv(image)
        updateAngelAndThrottle(speed)
        send_control(calculate_steeringAngle(dist_left, dist_right), throttle)

        # Code for the snake line driving
        # if COUNTER % 2 == 0:
        #     updateAngelAndThrottle(speed)
        #
        # send_control(angle, throttle)
        #
        # COUNTER = COUNTER + 1
    else:
        socket_io.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    socket_io.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

def calculate_steeringAngle(dist_left, dist_right):
    left_edge = -150
    right_edge = 150
    # 150 - 100 = 50 / 200 = 0.25 -> 100 would be the distance to have the max angle of 0.25 (7°)
    dividend = 200
    angle = 0

    if dist_right is not None and dist_right < right_edge:
        angle = -((right_edge - dist_right) / dividend)
    elif dist_left is not None and dist_left > left_edge:
        angle = ((left_edge * (-1)) - (dist_left * (-1))) / dividend
    return angle

def lane_detection_opencv(image):
    mask = cv2.imread("data/mask_sim.png", cv2.IMREAD_GRAYSCALE)
    ksize = (4, 4)
    blurImg = cv2.blur(image, ksize)
    #processedImage, dist_left, dist_right = processFrame(blurImg, mask)
    processedImage, dist_left, dist_right = our_process_frame(blurImg, mask)
    cv2.imshow("detected filtered lines", processedImage)
    cv2.waitKey(1)
    return dist_left, dist_right

if __name__ == '__main__':
    socket_io.run(app, port=4567)
