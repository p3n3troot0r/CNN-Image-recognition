import numpy as np
from scipy.ndimage import imread

# Prefix for hard drive with data on it. Change this depending on your data location and drive config for OS, etc.
# DATA_DRIVE_PREFIX = "H:"
DATA_DRIVE_PREFIX = "/media/neil/Neil's 250GB HDD"

# Directories for input image data to preprocess and grid split
# IMAGE_DIR = DATA_DRIVE_PREFIX + "/ImageNet_ILSVRC2015_SceneClassification_Small/data/val/"
IMAGE_DIR = DATA_DRIVE_PREFIX + "/ImageNet_ILSVRC2015_SceneClassification_Small/compressed_data/val/"

IMAGE_LIST_PATH = "/ImageNet_ILSVRC2015_SceneClassification_Small/Places2_devkit/data/val.txt"

PATCH_LENGTH = 32

def grid_split(arr):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//PATCH_LENGTH, PATCH_LENGTH, -1, PATCH_LENGTH)
               .swapaxes(1,2)
               .reshape(-1, PATCH_LENGTH, PATCH_LENGTH))


image_list = open(DATA_DRIVE_PREFIX + IMAGE_LIST_PATH).read().splitlines()

num_grid_blocks_per_image = (256/PATCH_LENGTH)**2
num_new_images = num_grid_blocks_per_image * len(image_list)
new_image_list = np.zeros((num_new_images, PATCH_LENGTH, PATCH_LENGTH))

for i in range(len(image_list)):
    print i, '/', len(image_list)
    # load image from list (for our data, we need to remove the space-separated category number from the list item)
    # also, convert it to greyscale while we're here using imread's flatten flag
    image = imread(IMAGE_DIR + image_list[i].split(" ")[0], flatten=True)

    image_grid_split = grid_split(image)

    new_image_list[i*num_grid_blocks_per_image : (i+1)*num_grid_blocks_per_image] = image_grid_split

np.random.seed(9000)
np.random.shuffle(new_image_list)

NUM_TEST_IMAGES = len(new_image_list) / 32  # Should be 41000
NUM_TRAIN_IMAGES = len(new_image_list) - NUM_TEST_IMAGES    # Should be 1312000 - 41000 = 1271000
NUM_TRAIN_IMAGES_SMALL_SET = 300000
NUM_TEST_IMAGES_SMALL_SET = 20000

FILE_PREFIX = "compressed_"

# np.save(FILE_PREFIX + "patches_train_full_" + str(NUM_TRAIN_IMAGES) + ".npy", arr=new_image_list[0 : NUM_TRAIN_IMAGES])
np.save(FILE_PREFIX + "patches_test_full_" + str(NUM_TEST_IMAGES) + ".npy", arr=new_image_list[NUM_TRAIN_IMAGES : NUM_TRAIN_IMAGES + NUM_TEST_IMAGES])
# np.save(FILE_PREFIX + "patches_train_small_" + str(NUM_TRAIN_IMAGES_SMALL_SET) + ".npy", arr=new_image_list[0 : NUM_TRAIN_IMAGES_SMALL_SET])
np.save(FILE_PREFIX + "patches_test_small_" + str(NUM_TEST_IMAGES_SMALL_SET) + ".npy", arr=new_image_list[NUM_TRAIN_IMAGES : NUM_TRAIN_IMAGES + NUM_TEST_IMAGES_SMALL_SET])
