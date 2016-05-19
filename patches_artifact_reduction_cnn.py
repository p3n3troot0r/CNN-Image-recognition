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

TRAIN_MODEL = True
LOAD_PRETRAINED_MODEL = True
PRETRAINED_MODEL_FILE = './weights/weights_400e_val_loss_0.5283.hdf5.hdf5'

PATCH_LENGTH = 32

# Data Prep
data_train = np.load('uncompressed_patches_train_small_300000.npy')
data_test = np.load('uncompressed_patches_test_small_20000.npy')[:5000]
compressed_train = np.load('compressed_patches_train_small_300000.npy')
compressed_test = np.load('compressed_patches_test_small_20000.npy')[:5000]

data_train = data_train.astype('float32') / 255.
data_test = data_test.astype('float32') / 255.
data_train = np.reshape(data_train, (len(data_train), 1, PATCH_LENGTH, PATCH_LENGTH))
data_test = np.reshape(data_test, (len(data_test), 1, PATCH_LENGTH, PATCH_LENGTH))

compressed_train = compressed_train.astype('float32') / 255.
compressed_test = compressed_test.astype('float32') / 255.
compressed_train = np.reshape(compressed_train, (len(compressed_train), 1, PATCH_LENGTH, PATCH_LENGTH))
compressed_test = np.reshape(compressed_test, (len(compressed_test), 1, PATCH_LENGTH, PATCH_LENGTH))

# CNN Prep
input_img = Input(shape=(1, 32, 32))

conv1 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(input_img)
pool1 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv1)
conv2 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv2)

conv3   = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool2)
upsamp1 = UpSampling2D(size=(2, 2))(conv3)
conv4   = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(upsamp1)
upsamp2 = UpSampling2D(size=(2, 2))(conv4)

output_img = Convolution2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(upsamp2)

model = Model(input_img, output_img)

if LOAD_PRETRAINED_MODEL:
    model.load_weights(PRETRAINED_MODEL_FILE)

if TRAIN_MODEL:
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy')
    model.fit(compressed_train, data_train,
                    nb_epoch=1000,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(compressed_test, data_test),
                    callbacks=[TensorBoard(log_dir='/tmp/arcnn', histogram_freq=1, write_graph=True), ModelCheckpoint(filepath="./tmp_weights/weights_pt2.{epoch:02d}-{val_loss:.4f}.hdf5")])

# Test prediction
decoded_imgs = model.predict(compressed_test)

# Display Results
n = 10
plt.figure(figsize=(20, 4))
for im in range(1, n):
    i = random.randint(0, len(decoded_imgs))
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(data_test[i].reshape(PATCH_LENGTH, PATCH_LENGTH))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display compressed
    ax = plt.subplot(3, n, i + n)
    plt.imshow(compressed_test[i].reshape(PATCH_LENGTH, PATCH_LENGTH))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n*2)
    plt.imshow(decoded_imgs[i].reshape(PATCH_LENGTH, PATCH_LENGTH))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
