import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask
from flask_socketio import SocketIO
import keras.models
import edgeDetection.behavioral_cloning_utils as utils

app = Flask(__name__)
socket_io = SocketIO(app)

model = None


@socket_io.on('telemetry')
def telemetry(data):
    if data:
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        image = np.asarray(image)  # from PIL image to numpy array
        image = utils.preprocess(image)  # apply preprocessing
        image = np.array([image])  # model expects batch of images (4D array)

        # predict the steering angle for the image
        steering_angle = float(model.predict(image, batch_size=1))

        throttle = 1 if speed < 20 else 0

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
    model = keras.models.load_model('model-001.h5')
    model.summary()
    socket_io.run(app, port=4567)
