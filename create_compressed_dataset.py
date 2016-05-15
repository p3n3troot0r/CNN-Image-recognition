import os
from PIL import Image
import time

# JPEG compression image qualities (1-100) (default: 25)
jpeg_quality = 25
# Prefix for hard drive with data on it. Change this depending on your data location and drive config for OS, etc.
data_drive_prefix = "H:"

# Directories for input image data, and output compressed image data
data_dir = data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/data/"
compressed_data_dir = data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/compressed_data/"

for subdir, dirs, files in os.walk(data_dir):
    print 'Collecting files to compress (may take a while)...'
    start_time = time.time()
    i = 1
    for file in files:
        if i == 1:
            print "File collection done in", (time.time() - start_time)/60.0, "minutes."
            start_time = time.time()

        print str(i) + '/' + str(len(files)) + ' - ' + file
        image = Image.open(subdir + '/' + file)

        new_subdir = compressed_data_dir + subdir[len(data_dir):]
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        image.save(new_subdir + '/' + file, "JPEG", quality=jpeg_quality)
        i += 1
    print "Batch compression of current subdir done in", (time.time() - start_time)/60.0, "minutes."
