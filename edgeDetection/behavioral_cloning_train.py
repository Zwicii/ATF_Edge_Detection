import pandas as pd
import numpy as np
import sklearn
import keras
from edgeDetection.behavioral_cloning_utils import INPUT_SHAPE, batch_generator
import os
import sklearn.model_selection

# for debugging, allows reproducible (deterministic) results
np.random.seed(0)


def load_data():
    # read CSV file into a single data frame variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), 'data/TrainingData', 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # camera images -> input data
    input = data_df[['center', 'left', 'right']].values
    # steering commands -> output data
    output = data_df['steering'].values

    # split the data into a training (80%) and testing (20%)
    input_train, input_test, output_train, output_test = sklearn.model_selection.train_test_split(input, output,
                                                                                                  test_size=0.2,
                                                                                                  random_state=0)

    return input_train, input_test, output_train, output_test


def build_model():
    model = keras.models.Sequential(name="my-self-driving-car")
    model.add(keras.layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))  # pixel values between -1 and 1
    # TODO: add some convolutional layers to the model
    model.add(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=2, activation="elu"))
    model.add(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=2, activation="elu"))
    model.add(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=2, activation="elu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="elu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="elu"))

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    # TODO: add some fully connected layers to the model
    model.add(keras.layers.Dense(100, activation="elu"))
    model.add(keras.layers.Dense(50, activation="elu"))
    model.add(keras.layers.Dense(10, activation="elu"))

    model.add(keras.layers.Dense(1))  # output layer
    model.summary()

    return model


def train_model(model, input_train, input_test, output_train, output_test):
    # save the model after every epoch
    checkpoint = keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5',
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=False,
                                                 mode='auto')

    # use mean squared error between expected steering angle and actual steering angle for gradient decent optimization
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=1.0e-4))

    # fit model using batch generator
    # 'data' specifies the folder for our training data
    model.fit(batch_generator('data/TrainingData', input_train, output_train, 40, True),
              steps_per_epoch=20000,
              epochs=5,
              max_queue_size=10,
              validation_data=batch_generator('data/TrainingData', input_test, output_test, 40, False),
              validation_steps=len(input_test),
              callbacks=[checkpoint],
              verbose=1)

if __name__ == '__main__':
    data = load_data()
    model = build_model()
    train_model(model, *data)
