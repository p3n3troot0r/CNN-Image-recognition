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

DEBUG = False

# Prefix for hard drive with data on it. Change this depending on your data location and drive config for OS, etc.
# data_drive_prefix = "H:"
data_drive_prefix = "/media/neil/Neil's 250GB HDD"

# Directories for input image data, and output compressed image data
data_dir = data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/data/"
compressed_data_dir = data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/compressed_data/"

random.seed(9000)
print "Collecting/shuffling image train/val image lists..."
train_image_list = open(data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/Places2_devkit/data/train.txt").read().splitlines()
random.shuffle(train_image_list)
val_image_list = open(data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/Places2_devkit/data/val.txt").read().splitlines()
random.shuffle(val_image_list)

def get_preproc_data(batch_size, set):
    residual_data = np.zeros((batch_size, 256, 256))
    compressed_data = np.zeros((batch_size, 256, 256))
    start_time = time.time()
    for i in range(batch_size):
        if set == 'train':
            image_name = train_image_list[-1].split(" ")[0]
            del train_image_list[-1]
        elif set == 'val':
            image_name = val_image_list[-1].split(" ")[0]
            del val_image_list[-1]
        else:
            print "ERROR: Invalid set type"
            exit(-1)
        compressed_data[i] = imread(compressed_data_dir + set + '/' + image_name, flatten=True)
        # residual_data[i] = imread(data_dir + set + '/' + image_name, flatten=True) - compressed_data[i]
        residual_data[i] = imread(data_dir + set + '/' + image_name, flatten=True)
        if time.time() - start_time > 5.0:
            print str(i) + '/' + str(batch_size) + ' - ' + image_name
            start_time = time.time()
            if DEBUG:
                plt.close()
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,10))

                ax1.imshow(compressed_data[i], cmap='gray')
                ax1.set_title('Compressed Image')

                cax2 = ax2.imshow(residual_data[i], extent=[0, 1, 0, 1], cmap='gray')
                cbar = plt.colorbar(cax2)
                ax2.set_title('Residual Image')

                plt.tight_layout()
                plt.show()


    residual_data = residual_data.astype('float32') / 255.
    compressed_data = compressed_data.astype('float32') / 255.
    return residual_data, compressed_data

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


autoencoder = Sequential()
autoencoder.add(Convolution2D(nb_filter=128, nb_row=9, nb_col=9, border_mode='same', init='glorot_normal', input_shape=(1, 256, 256)))
autoencoder.add(BatchNormalization())
autoencoder.add(Activation('relu'))
autoencoder.add(MaxPooling2D((2, 2), border_mode='same'))
autoencoder.add(Dropout(0.5))
autoencoder.add(Convolution2D(nb_filter=64, nb_row=1, nb_col=1, border_mode='same', init='glorot_normal'))
autoencoder.add(BatchNormalization())
autoencoder.add(Activation('relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Dropout(0.5))
autoencoder.add(Convolution2D(nb_filter=1, nb_row=5, nb_col=5, border_mode='same', init='glorot_normal'))
autoencoder.add(BatchNormalization())
autoencoder.add(Activation('sigmoid'))

# input_img = Input(shape=(1, 256, 256))
#
# conv1 = Convolution2D(nb_filter=64, nb_row=9, nb_col=9, activation='relu', border_mode='same', init='glorot_normal')(input_img)
# conv2 = Convolution2D(nb_filter=32, nb_row=1, nb_col=1, activation='relu', border_mode='same', init='glorot_normal')(conv1)
# recon = Convolution2D(nb_filter=1, nb_row=5, nb_col=5, activation='relu', border_mode='same', init='glorot_normal')(conv2)

# x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(input_img)
# x = MaxPooling2D((2, 2), border_mode='same')(x)
# x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
# encoded = MaxPooling2D((2, 2), border_mode='same')(x)
#
# # at this point the representation is (8, 4, 4) i.e. 128-dimensional
#
# x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
# x = UpSampling2D((2, 2))(x)
# recon = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)

# autoencoder = Model(input_img, recon)
autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy')

residual_data_val, compressed_data_val = get_preproc_data(20, 'val')
residual_data_val = np.reshape(residual_data_val, (len(residual_data_val), 1, 256, 256))
compressed_data_val = np.reshape(compressed_data_val, (len(compressed_data_val), 1, 256, 256))

load_batch_size = 200

# autoencoder.load_weights(filepath="weights.104-1.03.hdf5")

for i in range(4000):
    residual_data, compressed_data = get_preproc_data(load_batch_size, 'val')
    residual_data = np.reshape(residual_data, (len(residual_data), 1, 256, 256))
    compressed_data = np.reshape(compressed_data, (len(compressed_data), 1, 256, 256))
    compressed_data_mean = np.mean(compressed_data)
    compressed_data_norm = compressed_data - compressed_data_mean
    # for j in range(load_batch_size):
    #     residual_data_grid = blockshaped(residual_data[j], 32, 32)
    #     compressed_data_grid = blockshaped(compressed_data[j], 32, 32)
    #     residual_data_grid = np.reshape(residual_data_grid, (len(residual_data_grid), 1, 32, 32))
    #     compressed_data_grid = np.reshape(compressed_data_grid, (len(compressed_data_grid), 1, 32, 32))
    #     autoencoder.fit(compressed_data_grid, residual_data_grid,
    #             nb_epoch=1,
    #             batch_size=8,
    #             shuffle=True,
    #             callbacks=[TensorBoard(log_dir='/tmp/arcnn'), ModelCheckpoint(filepath="weights.{epoch:02d}.hdf5", verbose=0, save_best_only=False, mode='auto')])

    autoencoder.fit(compressed_data_norm, residual_data,
                nb_epoch=10,
                batch_size=1,
                shuffle=True,
                validation_data=(compressed_data_val, residual_data_val),
                callbacks=[TensorBoard(log_dir='/tmp/arcnn'), ModelCheckpoint(filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, mode='auto')])