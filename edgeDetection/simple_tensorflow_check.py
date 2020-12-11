import keras
import cv2
import numpy as np

# TODO: install the packages with the following command
# TODO: pip install opencv-python pillow flask-socketio gevent-websocket tensorflow==2.1.0 keras==2.3 matplotlib pandas sklearn

if __name__ == '__main__':
    # build model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation=keras.activations.relu))
    model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

    # show details
    model.summary()

    # set optimizer and loss to be optimized
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam())

    # load MNIST dataset for training and testing
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # run training on training dataset
    model.fit(x=x_train, y=y_train, epochs=2)

    # test resulting model using first 10 images of test dataset
    for i in range(10):
        img = x_test[i]
        img_show = cv2.resize(img, None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST)
        img_input = np.expand_dims(img, axis=0)

        output = model.predict(img_input)
        score = int(np.max(output) * 100)
        result = np.argmax(output)

        cv2.imshow("Prediction: " + str(result) + ", Score: " + str(score) + "%", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
