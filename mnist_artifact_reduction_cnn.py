import numpy as np
import os
import random
import time
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist

LOAD_PRETRAINED_MODEL = True
# PRETRAINED_MODEL_FILE = './MNIST_weights/MNIST_weights_100e/weights.99-0.07.hdf5'
PRETRAINED_MODEL_FILE = './MNIST_weights/MNIST_weights_100e/weights.00-0.08.hdf5'

def load_compressed_data(x_train, x_test):
    y_train = np.zeros(shape=x_train.shape)
    y_test = np.zeros(shape=x_test.shape)
    for i in range(len(x_train)):
        y_train[i] = imread('./MNIST_compressed/train/' + str(i) + '.jpg', flatten=True)
    for i in range(len(x_test)):
        y_test[i] = imread('./MNIST_compressed/test/' + str(i) + '.jpg', flatten=True)
    return y_train, y_test


(data_train, _), (data_test, _) = mnist.load_data()
# y_train, y_test = load_compressed_data(data_train, data_test)
compressed_train = np.load('./MNIST_compressed/compressed_train.npz')['compressed_train']
compressed_test = np.load('./MNIST_compressed/compressed_test.npz')['compressed_test']

data_train = data_train.astype('float32') / 255.
data_test = data_test.astype('float32') / 255.
data_train = np.reshape(data_train, (len(data_train), 1, 28, 28))
data_test = np.reshape(data_test, (len(data_test), 1, 28, 28))

compressed_train = compressed_train.astype('float32') / 255.
compressed_test = compressed_test.astype('float32') / 255.
compressed_train = np.reshape(compressed_train, (len(compressed_train), 1, 28, 28))
compressed_test = np.reshape(compressed_test, (len(compressed_test), 1, 28, 28))

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (32, 7, 7)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)


autoencoder = Model(input_img, decoded)

if LOAD_PRETRAINED_MODEL:
    autoencoder.load_weights(PRETRAINED_MODEL_FILE)
else:
    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy')
    autoencoder.fit(compressed_train, data_train,
                    nb_epoch=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(compressed_test, data_test),
                    callbacks=[TensorBoard(log_dir='/tmp/arcnn', histogram_freq=1, write_graph=True), ModelCheckpoint(filepath="./MNIST_weights/weights.{epoch:02d}-{val_loss:.4f}.hdf5")])

decoded_imgs = autoencoder.predict(compressed_test)

# Display Results
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(data_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display compressed
    ax = plt.subplot(3, n, i + n)
    plt.imshow(compressed_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n*2)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()