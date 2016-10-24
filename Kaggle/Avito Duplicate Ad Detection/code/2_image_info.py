#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Mikel
#### Avito Duplicate Ad Detection
# 2_image_info.py
# Creates a database of images and metadata about them, including dHash

import numpy as np
import pandas as pd
import cv2
import feather
import glob
import sys
import time
import os
import gc
from multiprocessing import Pool
from PIL import Image
from collections import Counter

import libavito as a

print(a.c.BOLD + 'Generating image info ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '2_image_info.py')

## Read settings required by script
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc
debug = config.debug
root = config.images_root

# Function to compute difference hash of image
def DifferenceHash(img):
    theImage = Image.fromarray(img)
    # Convert the image to 8-bit grayscale.
    theImage = theImage.convert("L")  # 8-bit grayscale
    # Squeeze it down to an 8x8 image.
    theImage = theImage.resize((8, 8), Image.ANTIALIAS)
    # Go through the image pixel by pixel.
    # Return 1-bits when a pixel is equal to or brighter than the previous
    # pixel, and 0-bits when it's below.
    # Use the 64th pixel as the 0th pixel.
    previousPixel = theImage.getpixel((0, 7))
    differenceHash = 0
    for row in range(0, 8, 2):
        # Go left to right on odd rows.
        for col in range(8):
            differenceHash <<= 1
            pixel = theImage.getpixel((col, row))
            differenceHash |= 1 * (pixel >= previousPixel)
            previousPixel = pixel
        row += 1
        # Go right to left on even rows.
        for col in range(7, -1, -1):
            differenceHash <<= 1
            pixel = theImage.getpixel((col, row))
            differenceHash |= 1 * (pixel >= previousPixel)
            previousPixel = pixel
    return differenceHash

def get_info(file_loc):
    try:
        # Get size of image
        size = os.path.getsize(file_loc)

        # Attempt to load image
        img = cv2.imread(file_loc)
        try:
            # Test if image is corrupt
            assert img.shape[0] * img.shape[1] > 0
        except:
            print('[WARNING] Image ' + file_loc + ' is corrupt, skipping.')
            raise

        # Get image metadata
        width = img.shape[1]
        height = img.shape[0]

        # Get ratio of image dimensions
        ratio = round(min(width, height) / max(width, height), 2)

        # Compute difference hash of image and convert to hex
        dhash = '%(hash)016x' % {"hash": DifferenceHash(img)}

        return [width, height, ratio, dhash, size]

    except KeyboardInterrupt:
        raise
    except:
        print('[WARNING] Image ' + file_loc + ' failed to process.')
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

def process_line(f):
    # Get image ID
    img_id = f.split('/')[-1].split('.')[0]
    # Retrieve info for image
    d = get_info(f)
    # Construct list and return
    info = []
    info.append(img_id)
    info.extend(d)
    return info

# Recursively glob for jpeg files in the image root
start = time.time()
print('Looking for images in ' + root + ' ... ', end='', flush=True)
files = glob.glob(root + '**/*.jpg', recursive=True)
a.print_elapsed(start)

print('Found ' + str(len(files)) + ' images.')

l_id = []
l_width = []
l_height = []
l_ratio = []
l_hash = []
l_size = []
o = len(files)
if nthreads == 1:
    print('Extracting image info with 1 thread ...')
    k = 0
    # Iterate over files
    for f in files:
        x = process_line(f)
        l_id.append(x[0])
        l_width.append(x[1])
        l_height.append(x[2])
        l_ratio.append(x[3])
        l_hash.append(x[4])
        l_size.append(x[5])
        k += 1
        if k % 1000 == 0:
            a.print_progress(k, start, o)
# Otherwise perform multi-threaded mapping
else:
    print('Extracting image info multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads)
    newdata = pool.map(process_line, files)
    pool.close()
    for x in newdata:
        l_id.append(x[0])
        l_width.append(x[1])
        l_height.append(x[2])
        l_ratio.append(x[3])
        l_hash.append(x[4])
        l_size.append(x[5])
    del newdata
    gc.collect()

    a.print_elapsed(start)

print('Finding hash-counts ...', end='', flush=True)
start = time.time()
counttable = Counter(l_hash)
l_hashcount = []
for h in l_hash:
    l_hashcount.append(counttable[h])
a.print_elapsed(start)

# Bind lists to dataframe
df = pd.DataFrame()
df['image'] = l_id
df['width'] = l_width
df['height'] = l_height
df['ratioOfDimension'] = l_ratio
df['imagehash'] = l_hash
df['FreqOfHash'] = l_hashcount
df['imagesize'] = l_size

start = time.time()
print('Caching image data ... ', end='', flush=True)

# Save updated dataset
feather.write_dataframe(df, cache_loc + 'image_database.fthr')
df.to_csv(cache_loc + 'image_database.csv', index=False)

a.print_elapsed(start)
print('Image info extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('image_info_OK\n')
f.close()
