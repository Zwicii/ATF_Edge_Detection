import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import random
from PIL import Image

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from file
    """
    return mpimg.imread(os.path.join(data_dir, "IMG", os.path.basename(image_file)))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize image to input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert image from RGB to YUV (this is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    randNum = random.randint(0, 2)
    img = None

    if randNum == 0:
        img = center
    elif randNum == 1:
        img = left
        steering_angle += 0.2
    else:
        img = right
        steering_angle -= 0.2

    return load_image(data_dir, img), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right and adjust the steering angle.
    """
    randNum = random.randint(0, 1)

    if randNum == 0:
        steering_angle *= -1
        image = np.fliplr(image)

    return image, steering_angle


def random_translate(image, steering_angle):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    h, w = image.shape[:2]
    th = random.randint(-50, 50)
    tv = random.randint(-5, 5)

    T = np.float32([[1, 0, th], [0, 1, tv]])

    steering_angle = steering_angle + th * 0.002
    image = cv2.warpAffine(image, T, (w, h))

    return image, steering_angle


def augment(data_dir, center, left, right, steering_angle):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, use_augmentation):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]

            if use_augmentation and np.random.rand() < 0.6:
                # data augmentation
                image, steering_angle = augment(data_dir, center, left, right, steering_angle)
            else:
                # standard data
                image = load_image(data_dir, center)

            # add image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


if __name__ == '__main__':
    print("Testing the augmentation methods...")
    image, steering_angle = choose_image('data/TrainingData', 'center_2020_12_11_16_16_13_693.jpg', 'left_2020_12_11_16_16_13_693.jpg', 'right_2020_12_11_16_16_13_693.jpg', 0.2)
    cv2.imshow("chosen image - steer: " + str(steering_angle), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image, steering_angle = random_flip(image, steering_angle)
    cv2.imshow("flipped image - steer: " + str(steering_angle), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image, steering_angle = random_translate(image, steering_angle)
    cv2.imshow("shifted image - steer: " + str(steering_angle), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
