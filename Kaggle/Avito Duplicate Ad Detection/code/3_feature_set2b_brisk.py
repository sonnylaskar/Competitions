#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set2b_brisk.py
# Extracts BRISK keypoints/descriptors from images and uses them to compute features

import numpy as np
import pandas as pd
import cv2
import feather
import glob
import sys
import os
import gc
import time
import warnings
from random import random
from multiprocessing import Pool

import libavito as a

# Set this to true to write any errors and current progress in the cache folder
write_info = False

def find_brisk_features(row):
    row[2] = str(row[2])
    row[3] = str(row[3])
    x_imgs = row[2].split(', ')
    y_imgs = row[3].split(', ')

    if write_info is True:
        p = open(cache_loc + 'brisk_progress.txt', 'a')
        p.write('a\n')
        p.close()

    bfmean_distances = []
    kb_distances = []
    bfmed_distances = []

    # Skip if no images present
    if x_imgs not in [['nan'], ['None']] and y_imgs not in [['nan'], ['None']]:
        for x in x_imgs:

            # Get file location from DB
            #print(x)
            f = image_db[x]

            oserror = 0
            try:
                img = cv2.imread(f, cv2.IMREAD_COLOR)
            except Exception as e:
                oserror = 1
                print(str(e) + " FILE ERROR at " + str(x) + " !!!")
                if write_info is True:
                    p = open(cache_loc + 'brisk_errors_file.txt', 'a')
                    p.write('FILE ERROR at ' + str(x) + "\n")
                    p.close()

            if oserror == 0:
                try:
                    _, hx = brisk.detectAndCompute(img,None)
                except Exception as e:
                    hx = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                    print(str(e) + " BRISK ERROR at " + str(x) + " !!!")
                    if write_info is True:
                        p = open(cache_loc + 'brisk_errors_extract.txt', 'a')
                        p.write('BRISK ERROR at ' + str(x) + "\n")
                        p.close()

                for y in y_imgs:

                    f = image_db[y]

                    oserror = 0
                    try:
                        img = cv2.imread(f, cv2.IMREAD_COLOR)
                    except Exception as e:
                        oserror = 1
                        print(str(e) + " FILE ERROR at " + str(y) + " !!!")
                        if write_info is True:
                            p = open(cache_loc + 'brisk_errors_file.txt', 'a')
                            p.write('FILE ERROR at ' + str(x) + "\n")
                            p.close()

                    if oserror == 0:
                        try:
                            _, hy = brisk.detectAndCompute(img,None)
                        except Exception as e:
                            hy = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                            print(str(e) + " BRISK ERROR at " + str(y) + " !!!")
                            if write_info is True:
                                p = open(cache_loc + 'brisk_errors_extract.txt', 'a')
                                p.write('BRISK ERROR at ' + str(x) + "\n")
                                p.close()

                        try:
                            # Cross-check matching
                            #with suppress_stdout_stderr():
                            bf_matches = bf.match(hx, hy)
                            #bf_matches = sorted(bf_matches, key=lambda x:x.distance)
                            bf_matchdists = []
                            for m in bf_matches:
                                bf_matchdists.append(m.distance)
                            bfmean_distances.append(np.nanmean(bf_matchdists))
                            bfmed_distances.append(np.nanmedian(bf_matchdists))

                            # # KNN matching
                            # #kb_matches = bf.knnMatch(hx, hy, k=20)
                            # kb_matches = sorted(bf_matches, key=lambda x:x.distance)[:20]
                            # print(kb_matches)
                            # kb_matchdists = []
                            # for m in kb_matches:
                            #     kb_matchdists.append(m.distance)
                            #
                            # kb_distances.append(np.nanmean(kb_matchdists))

                        except Exception as e:
                            #print(str(e) + " MATCH ERROR at " + str(x) + ' ' + str(y) + " !!!")
                            if write_info is True:
                                p = open(cache_loc + 'brisk_errors_matching.txt', 'a')
                                p.write('MATCH ERROR at ' + str(x) + ' ' + str(y) + "\n")
                                p.close()

    bfmean_mindist = np.nan
    bfmean_meddist = np.nan
    bfmean_propzero = np.nan
    bfmean_prop80 = np.nan
    bfmean_prop30 = np.nan
    bfmed_mindist = np.nan
    bfmed_meddist = np.nan
    bfmed_propzero = np.nan
    bfmed_prop80 = np.nan
    bfmed_prop30 = np.nan
    # kb20_mindist = np.nan
    # kb20_meddist = np.nan
    # kb20_propzero = np.nan
    # kb20_prop80 = np.nan
    # kb20_prop30 = np.nan

    if len(bfmean_distances) != 0:
        bfmean_mindist = np.nanmin(bfmean_distances)
        bfmean_meddist = np.nanmedian(bfmean_distances)
        bfmean_propzero = (np.array(bfmean_distances) == 0).astype(int).mean()
        bfmean_prop80 = (np.array(bfmean_distances) < 80).astype(int).mean()
        bfmean_prop30 = (np.array(bfmean_distances) < 30).astype(int).mean()

    if len(bfmed_distances) != 0:
        bfmed_mindist = np.nanmin(bfmed_distances)
        bfmed_meddist = np.nanmedian(bfmed_distances)
        bfmed_propzero = (np.array(bfmed_distances) == 0).astype(int).mean()
        bfmed_prop80 = (np.array(bfmed_distances) < 80).astype(int).mean()
        bfmed_prop30 = (np.array(bfmed_distances) < 30).astype(int).mean()

    # if len(kb_distances) != 0:
    #     kb20_mindist = np.nanmin(kb_distances)
    #     kb20_meddist = np.nanmedian(kb_distances)
    #     kb20_propzero = (np.array(kb_distances) == 0).astype(int).mean()
    #     kb20_prop80 = (np.array(kb_distances) < 80).astype(int).mean()
    #     kb20_prop30 = (np.array(kb_distances) < 30).astype(int).mean()

    try:
        # del distances
        del hx
        del hy
        # del matches
        # del matchdists
        del bfmean_distances
        del bfmed_distances
        del kb_distances
        del bf_matches
        del bf_matchdists
        # del kb_matches
        # del kb_matchdists
        del x_imgs
        del y_imgs
        del img
    except:
        pass

    # Sometimes run the garbage collector
    if random() > 0.99:
        gc.collect()

    return_val = [bfmean_mindist, bfmean_meddist, bfmean_propzero, bfmean_prop80, bfmean_prop30, bfmed_mindist, bfmed_meddist, bfmed_propzero, bfmed_prop80, bfmed_prop30]
    return return_val

print(a.c.BOLD + 'Extracting set2b BRISK features ...' + a.c.END)

# Define BRISK
brisk = cv2.BRISK_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Suppress expected warnings from pandas
warnings.filterwarnings("ignore", message='mean of empty slice|all-nan (axis|slice) encountered')

# Since OpenCV completely ignores suppressions, we have to use a context manager to silence any errors that originate from it.
class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)

    def close(self):
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def suppress_pool_init():
        # Open a pair of null files
        null_fds = [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Assign the null pointers to stdout and stderr.
        os.dup2(null_fds[0],1)
        os.dup2(null_fds[1],2)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set2b_brisk.py')

## Read settings required by script
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc
debug = config.debug
if mode == 0:
    root = config.train_images_root
    df = feather.read_dataframe(cache_loc + 'train.fthr')
if mode == 1:
    root = config.test_images_root
    df = feather.read_dataframe(cache_loc + 'test.fthr')

# Select columns required by script
df = df[['itemID_1', 'itemID_2', 'images_array_1', 'images_array_2']]
gc.collect()

# Recursively glob for jpeg files in the image root
start = time.time()
print('Looking for images in ' + root + ' ... ', end='', flush=True)
image_locs = glob.glob(root + '**/*.jpg', recursive=True)
a.print_elapsed(start)

print('Found ' + str(len(image_locs)) + ' images.')

# For each image found, get the image ID and store file/location in a dictionary
start = time.time()
print('Mapping image IDs to folder structure ... ', end='', flush=True)
image_db = {}
for loc in image_locs:
    iid = loc.split('/')[-1].split('.jpg')[0]  # Remove parent folders and extension
    #print(iid)
    image_db[iid] = loc
a.print_elapsed(start)

o = len(df.index)
if nthreads == 1:
    print('Extracting BRISK features with 1 thread ...')
    k = 0
    supressor = suppress_stdout_stderr()
    # Iterate over files
    ftrs = []
    for row in df.values:
        with supressor:
            x = find_brisk_features(row)
        ftrs.append(x)
        k += 1
        if k % 100 == 0:
            a.print_progress(k, start, o)

    supressor.close()
# Otherwise perform multi-threaded mapping
else:
    print('Extracting BRISK features multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads, initializer=suppress_pool_init)
    ftrs = pool.map(find_brisk_features, df.values)
    pool.close()
    gc.collect()

    a.print_elapsed(start)

start = time.time()
print('Parsing features ... ', end='', flush=True)

# Convert to dataframe and add feature names
df_ftrs = pd.DataFrame(ftrs)
df_ftrs.columns = ["set2b_" + c for c in ["bfmean_mindist", "bfmean_meddist", "bfmean_propzero", "bfmean_prop80", "bfmean_prop30", "bfmed_mindist", "bfmed_meddist", "bfmed_propzero", "bfmed_prop80", "bfmed_prop30"]]
df_ftrs['itemID_1'] = df['itemID_1']
df_ftrs['itemID_2'] = df['itemID_2']
a.print_elapsed(start)

start = time.time()
print('Caching data to disk ... ', end='', flush=True)
# Save updated dataset
if mode == 0:
    feather.write_dataframe(df_ftrs, cache_loc + 'features_train_set2b_brisk.fthr')
if mode == 1:
    feather.write_dataframe(df_ftrs, cache_loc + 'features_test_set2b_brisk.fthr')

a.print_elapsed(start)
print('set2b extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set2b_OK\n')
f.close()
