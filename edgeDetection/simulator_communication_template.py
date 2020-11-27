import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socket_io = SocketIO(app)


@socket_io.on('telemetry')
def telemetry(data):
    if data:
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # TODO: calculate steering angle and throttle from image and speed

        send_control(steering_angle, throttle)
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
