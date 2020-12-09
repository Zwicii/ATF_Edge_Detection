import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socket_io = SocketIO(app)


throttle = 0
angle = 0
step = -0.1
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
        step = -0.1
    elif angle <= -1:
        angle = 0
        step = 0.1



@socket_io.on('telemetry')
def telemetry(data):
    global COUNTER
    if data:
        speed = float(data["speed"])

        if COUNTER % 2 == 0:
            updateAngelAndThrottle(speed)

        send_control(angle, throttle)

        COUNTER = COUNTER + 1
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


if __name__ == '__main__':
    socket_io.run(app, port=4567)
