import os
from PIL import Image

# JPEG compression image qualities (1-100) (default: 25)
jpeg_quality = 25
# Prefix for hard drive with data on it. Change this depending on your data location and drive config for OS, etc.
data_drive_prefix = "H:"

# Directories for input image data, and output compressed image data
data_dir = data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/data/"
compressed_data_dir = data_drive_prefix + "/ImageNet_ILSVRC2015_SceneClassification_Small/compressed_data/"

print 'Collecting files to compress (may take a while)...'
for subdir, dirs, files in os.walk(data_dir):
    i = 0
    for file in files:
        print str(i) + '/' + str(len(files)) + ' - ' + file
        image = Image.open(subdir + '/' + file)

        new_subdir = compressed_data_dir + subdir[len(data_dir):]
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        image.save(new_subdir + '/' + file, "JPEG", quality=jpeg_quality)
        i += 1
