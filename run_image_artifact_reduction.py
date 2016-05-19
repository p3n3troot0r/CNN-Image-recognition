import numpy as np
import math
import os
import random
import time
from scipy.ndimage import imread
from scipy.misc import imsave
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

PRETRAINED_MODEL_FILE = "./weights/weights_400e_val_loss_0.5283.hdf5"
IMAGE_FILE = "/media/neil/Neil's 250GB HDD/ImageNet_ILSVRC2015_SceneClassification_Small/data/test/Places2_test_00000015.jpg"
# IMAGE_FILE = "/media/neil/Neil's 250GB HDD/Meme.jpg"

PRESERVE_COLOR = False
SAVE_IMAGE = False

def reduce_image_artifacts(image):
    # if the image is RGB and we want to preserve it,
    if (len(image.shape) == 3 and PRESERVE_COLOR):
        # Call the function on all channels, and return the combined image
        image_out = np.zeros(image.shape, 'uint8')
        image_out[..., 0] = reduce_image_artifacts(image[..., 0])
        image_out[..., 1] = reduce_image_artifacts(image[..., 1])
        image_out[..., 2] = reduce_image_artifacts(image[..., 2])
        return image_out

    # at this point we only want to work with grayscale images, convert if necessary
    if (len(image.shape) == 3):
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    # temporarily extend the image in the x and y directions so the width and height are divisible by 32
    # this will help prevent the CNN from zero-padding and introducing black artifacts around right/bottom edges
    extension_row = int((math.ceil(image.shape[0]*1.0 / 32) * 32) - image.shape[0])
    extension_col = int((math.ceil(image.shape[1]*1.0 / 32) * 32) - image.shape[1])
    image = np.lib.pad(image, ((0, extension_row), (0, extension_col)), 'reflect')

    # Reshape image to 1 x 1 x r x c, to satisfy keras (save original shape for later)
    image_r = image.shape[0]
    image_c = image.shape[1]
    image = np.reshape(image, (1, 1, image.shape[0], image.shape[1]))
    # Also convert to float32 and make pixel values 0-1 instead of 0-255
    image = image.astype('float32') / 255.


    # CNN Prep
    input_img = Input(shape=(1, image_r, image_c))

    conv1 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv1)
    conv2 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv2)

    conv3 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool2)
    upsamp1 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(upsamp1)
    upsamp2 = UpSampling2D(size=(2, 2))(conv4)

    output_img = Convolution2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(upsamp2)

    model = Model(input_img, output_img)
    model.load_weights(PRETRAINED_MODEL_FILE)

    image_out = model.predict(image)

    # Reshape output to match input and convert back to 0-255 pixel values
    image_out = image_out.reshape(image_r, image_c) * 255

    # Crop image to remove extensions we did earlier
    image_out = image_out[: image_out.shape[0] - extension_row, : image_out.shape[1] - extension_col]
    # Round and convert back to normal image type (uint8)
    image_out = image_out.round().astype('uint8')
    return image_out

def main():
    if PRESERVE_COLOR:
        image = imread(IMAGE_FILE)
    else:
        image = imread(IMAGE_FILE, flatten=True)
    image_out = reduce_image_artifacts(image)
    image_out = reduce_image_artifacts(image_out)

    plt.figure(figsize=(20, 10))
    # display original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Compressed")

    # display cleaned
    ax = plt.subplot(1, 2, 2)
    plt.imshow(image_out)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Cleaned")
    plt.show()

    if SAVE_IMAGE:
        imsave(IMAGE_FILE.split("/")[-1]+'.png', image_out)

    return

if __name__ == "__main__":
    main()