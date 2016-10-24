#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set2b_brisk.py
# Extracts histogram/hue features from images

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

# Some feature config
histdim_a = 16
histdim_b = 2
hsv_sat_thresh = 10
hsv_val_thresh_low = 10

# Set this to true to write any errors and current progress in the cache folder
write_info = False

def get_histogram(img):
    # Generate normal histogram with two different resolutions
    hist = cv2.calcHist([img], [0, 1, 2], None, [histdim_a, histdim_a, histdim_a], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist = hist.flatten()
    hist2 = cv2.calcHist([img], [0, 1, 2], None, [histdim_b, histdim_b, histdim_b],[0, 256, 0, 256, 0, 256])
    hist2 = hist2.flatten()
    cv2.normalize(hist2, hist2)

    return hist, hist2

def get_hsv(img):
    # Convert to HSV, and use a threshold to select pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv = pd.DataFrame(hsv)
    hsv = hsv.loc[hsv[1] > hsv_sat_thresh]
    hsv = hsv.loc[hsv[2] > hsv_val_thresh_low]
    hsv150 = hsv.loc[hsv[1] > 150]
    hsv150 = hsv150[0].values
    hsv = hsv[0].values
    return hsv, hsv150

def get_hsv_hist(img):
    img, img150 = get_hsv(img)
    hist = np.array(np.histogram(img, histdim_a, range=(0, 180), density=True)[0], ndmin=2, dtype=np.float32).T
    hist150 = np.array(np.histogram(img150, histdim_a, range=(0, 180), density=True)[0], ndmin=2, dtype=np.float32).T
    return hist, hist150

def load_data(x):
    f = image_db[x]
    try:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        try:
            hist, hist2 = get_histogram(img)
            hsv_hist, hsv_hist150 = get_hsv_hist(img)
        except Exception as e:
            err('hist', x, e)
            return None

    except Exception as e:
        err('file', x, e)
        return None

    return hist, hsv_hist, hist2, hsv_hist150

def compute_features(x_imgs, y_imgs, hist_1, hist_2, hsv_1, hsv_2):
    try:
        # Data store for features

        inter_list = []
        corr_list = []
        chisq_list = []
        hell_list = []
        hue_inter_list = []
        hue_corr_list = []

        maxcor_x = {}
        maxint_x = {}
        minhell_x = {}
        minchi_x = {}
        maxcor_y = {}
        maxint_y = {}
        minhell_y = {}
        minchi_y = {}
        hue_maxint_x = {}
        hue_maxint_y = {}
        hue_maxcor_x = {}
        hue_maxcor_y = {}
        for ix in hist_1.keys():
                maxcor_x[ix] = 0.0
                maxint_x[ix] = 0.0
                minhell_x[ix] = 1000.0
                minchi_x[ix] = 1000.0
                hue_maxint_x[ix] = 0.0
                hue_maxcor_x[ix] = 0.0
        for iy in hist_2.keys():
                maxcor_y[iy] = 0.0
                maxint_y[iy] = 0.0
                minhell_y[iy] = 1000.0
                minchi_y[iy] = 1000.0
                hue_maxint_y[iy] = 0.0
                hue_maxcor_y[iy] = 0.0

        ###### HIST FEATURES #####

        for x in x_imgs:
            hist_x = hist_1[x]
            hsv_x = hsv_1[x]
            if hist_x is not None:
                for y in y_imgs:
                    hist_y = hist_2[y]
                    hsv_y = hsv_2[y]
                    if hist_y is not None:
                        try:
                            # Get distance between histograms
                            inter = cv2.compareHist(hist_x,hist_y,cv2.HISTCMP_INTERSECT)
                            corr = cv2.compareHist(hist_x,hist_y,cv2.HISTCMP_CORREL)
                            chisq = cv2.compareHist(hist_x,hist_y,cv2.HISTCMP_CHISQR)
                            hell = cv2.compareHist(hist_x,hist_y,cv2.HISTCMP_BHATTACHARYYA)

                            hue_inter = cv2.compareHist(hsv_x, hsv_y, cv2.HISTCMP_INTERSECT)
                            hue_corr = cv2.compareHist(hsv_x, hsv_y, cv2.HISTCMP_CORREL)

                            inter_list.append(inter)
                            corr_list.append(corr)
                            chisq_list.append(chisq)
                            hell_list.append(hell)
                            hue_inter_list.append(hue_inter)
                            hue_corr_list.append(hue_corr)

                            # find the best values per image in 1
                            if corr > maxcor_x[ix]:
                                maxcor_x[ix] = corr
                            if hell < minhell_x[ix]:
                                minhell_x[ix] = hell
                            if chisq < minchi_x[ix]:
                                minchi_x[ix] = chisq
                            if inter > maxint_x[ix]:
                                maxint_x[ix] = inter
                            if hue_inter > hue_maxint_x[ix]:
                                hue_maxint_x[ix] = hue_inter
                            if hue_corr > hue_maxcor_x[ix]:
                                hue_maxcor_x[ix] = hue_corr
                            # find the best values per image in 2
                            if corr > maxcor_y[iy]:
                                    maxcor_y[iy] = corr
                            if hell < minhell_y[iy]:
                                    minhell_y[iy] = hell
                            if chisq < minchi_y[iy]:
                                    minchi_y[iy] = chisq
                            if inter > maxint_y[iy]:
                                    maxint_y[iy] = inter
                            if hue_inter > hue_maxint_y[iy]:
                                hue_maxint_y[iy] = hue_inter
                            if hue_corr > hue_maxcor_y[iy]:
                                hue_maxcor_y[iy] = hue_corr

                        except Exception as e:
                            err('ftr', x, e)

        # Best histogram match features
        max_inter = max(inter_list)
        max_corr = max(corr_list)
        min_chisq = min(chisq_list)
        min_hell = min(hell_list)
        max_hue_inter = max(hue_inter_list)
        max_hue_corr = max(hue_corr_list)
        min_hue_inter = min(hue_inter_list)
        min_hue_corr = min(hue_corr_list)

        min_corr = min(corr_list)

        # Find median of photos in smaller set, so that larger sets are not penalised.
        xmed = max(int(min(len(x_imgs), len(y_imgs)) / 2), 0)

        xmed_inter = np.mean(sorted(inter_list)[:xmed])
        xmed_corr = np.mean(sorted(corr_list)[:xmed])
        xmed_chisq = np.mean(sorted(chisq_list, reverse=True)[:xmed])
        xmed_hell = np.mean(sorted(hell_list, reverse=True)[:xmed])
        hue_xmed_inter = np.mean(sorted(hue_inter_list)[:xmed])
        hue_xmed_corr = np.mean(sorted(hue_corr_list)[:xmed])

        # Get best image in x and best image in y
        maxcor_x_mean = np.fromiter(iter(maxcor_x.values()), dtype=float).mean()
        maxcor_y_mean = np.fromiter(iter(maxcor_y.values()), dtype=float).mean()
        maxint_x_mean = np.fromiter(iter(maxint_x.values()), dtype=float).mean()
        maxint_y_mean = np.fromiter(iter(maxint_y.values()), dtype=float).mean()
        minhell_x_mean = np.fromiter(iter(minhell_x.values()), dtype=float).mean()
        minhell_y_mean = np.fromiter(iter(minhell_y.values()), dtype=float).mean()
        minchi_x_mean = np.fromiter(iter(minchi_x.values()), dtype=float).mean()
        minchi_y_mean = np.fromiter(iter(minchi_y.values()), dtype=float).mean()
        hue_maxcor_x_mean = np.fromiter(iter(hue_maxcor_x.values()), dtype=float).mean()
        hue_maxcor_y_mean = np.fromiter(iter(hue_maxcor_y.values()), dtype=float).mean()
        hue_maxint_x_mean = np.fromiter(iter(hue_maxint_x.values()), dtype=float).mean()
        hue_maxint_y_mean = np.fromiter(iter(hue_maxint_y.values()), dtype=float).mean()

        maxcor_mean_max = max(maxcor_x_mean, maxcor_y_mean)
        maxcor_mean_min = min(maxcor_x_mean, maxcor_y_mean)
        maxint_mean_max = max(maxint_x_mean, maxint_y_mean)
        maxint_mean_min = min(maxint_x_mean, maxint_y_mean)
        minhell_mean_max = max(minhell_x_mean, minhell_y_mean)
        minhell_mean_min = min(minhell_x_mean, minhell_y_mean)
        minchi_mean_max = max(minchi_x_mean, minchi_y_mean)
        minchi_mean_min = min(minchi_x_mean, minchi_y_mean)
        hue_maxcor_mean_max = max(hue_maxcor_x_mean, hue_maxcor_y_mean)
        hue_maxcor_mean_min = min(hue_maxcor_x_mean, hue_maxcor_y_mean)
        hue_maxint_mean_max = max(hue_maxint_x_mean, hue_maxint_y_mean)
        hue_maxint_mean_min = min(hue_maxint_x_mean, hue_maxint_y_mean)

        ##### END FEATURES #####

        return [max_inter, max_corr, min_chisq, min_hell, max_hue_inter, max_hue_corr, min_hue_inter, min_hue_corr, xmed_inter, xmed_corr, xmed_chisq, xmed_hell, hue_xmed_inter, hue_xmed_corr, maxcor_mean_max, maxcor_mean_min, maxint_mean_max, maxint_mean_min, minhell_mean_max, minhell_mean_min, minchi_mean_max, minchi_mean_min, hue_maxcor_mean_max, hue_maxcor_mean_min, hue_maxint_mean_max, hue_maxint_mean_min, min_corr]

    except Exception as e:
        print('[WARNING] Feature generation failed for row with ' + str(e) + ' at ' + str((x_imgs, y_imgs)))

        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

def find_hist_features(row):
    row[0] = str(row[0])
    row[1] = str(row[1])
    id1 = row[2]
    id2 = row[3]
    x_imgs = row[0].split(', ')
    y_imgs = row[1].split(', ')

    if write_info is True:
        p = open(cache_loc + 'hist_progress.txt', 'a')
        p.write('a\n')
        p.close()

    # Skip if no images present for this item
    if x_imgs not in [['None'], ['nan']] and y_imgs not in [['None'], ['nan']]:
        try:
            ###### LOAD IMAGES & CALCULATE HISTOGRAMS #####

            # dictionaries for image data
            hist_1 = {}
            hist_2 = {}
            hsv_1 = {}
            hsv_2 = {}
            hist2_1 = {}
            hist2_2 = {}
            hsv_1 = {}
            hsv_2 = {}
            hsv150_1 = {}
            hsv150_2 = {}

            # For each image, load images and compute histograms, before storing in dict.
            # This way, each image only has to be loaded/computed once.
            for x in x_imgs:
                hist_x, hsv_x, hist2_x, hsv150_x = load_data(x)
                hist_1[x] = hist_x
                hsv_1[x] = hsv_x
                hist2_1[x] = hist2_x
                hsv150_1[x] = hsv150_x

            for y in y_imgs:
                hist_y, hsv_y, hist2_y, hsv150_y = load_data(y)
                hist_2[y] = hist_y
                hsv_2[y] = hsv_y
                hist2_2[y] = hist2_y
                hsv150_2[y] = hsv150_y

            final_ftrs = []

            final_ftrs.append(id1)
            final_ftrs.append(id2)

            # Compute comparison features on both sets of histograms
            ftrs = compute_features(x_imgs, y_imgs, hist_1, hist_2, hsv_1, hsv_2)
            ftrs2 = compute_features(x_imgs, y_imgs, hist2_1, hist2_2, hsv150_1, hsv150_2)

            final_ftrs.extend(ftrs)
            final_ftrs.extend(ftrs2)

            return final_ftrs

        except Exception as e:
            print('[WARNING] Failed to process row:' + str(row))
            print(e)

    # If some failure occurred, just return NaNs as features
    return [id1, id2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

# Generic eror handler
def err(where, x, e):
    if where == 'file':
        print(str(e) + " FILE ERROR at " + str(x) + " !!!")
        if write_info is True:
            p = open(cache_loc + 'hist_errors_file.txt', 'a')
            p.write('FILE ERROR at ' + str(x) + "\n")
            p.close()
    if where == 'hist':
        print(str(e) + " HIST ERROR at " + str(x) + " !!!")
        if write_info is True:
            p = open(cache_loc + 'hist_errors_extract.txt', 'a')
            p.write('HIST ERROR at ' + str(x) + "\n")
            p.close()
    if where == 'ftr':
        print(str(e) + " FTR ERROR at " + str(x) + " !!!")
        if write_info is True:
            p = open(cache_loc + 'hist_errors_extract.txt', 'a')
            p.write('FTR ERROR at ' + str(x) + "\n")
            p.close()

print(a.c.BOLD + 'Extracting set2c image histogram/hue features ...' + a.c.END)

# Suppress expected warnings from pandas
warnings.filterwarnings("ignore", message='mean of empty slice|all-nan (axis|slice) encountered')

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set2c_hist.py')

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
df = df[['images_array_1', 'images_array_2', 'itemID_1', 'itemID_2']]
gc.collect()

# Recursively glob for jpeg files in the image root
start = time.time()
print('Looking for images in ' + root + ' ... ', end='', flush=True)
image_locs = glob.glob(root + '**/*.jpg', recursive=False)
a.print_elapsed(start)

print('Found ' + str(len(image_locs)) + ' images.')

# For each image found, get the image ID and store file/location in a dictionary
start = time.time()
print('Mapping image IDs to folder structure ... ', end='', flush=True)
image_db = {}
for loc in image_locs:
    iid = loc.split('/')[-1].split('.jpg')[0]  # Remove parent folders and extension
    image_db[iid] = loc
a.print_elapsed(start)

o = len(df.index)
if nthreads == 1:
    print('Extracting features with 1 thread ...')
    k = 0
    # Iterate over rows
    vals = []
    for row in df.values:
        x = find_hist_features(row)
        vals.append(x)
        k += 1
        if k % 100 == 0:
            a.print_progress(k, start, o)
# Otherwise perform multi-threaded mapping
else:
    print('Extracting features multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads)
    vals = pool.map(find_hist_features, df.values)
    pool.close()
    gc.collect()

    a.print_elapsed(start)

start = time.time()
print('Parsing features ... ', end='', flush=True)
vals = pd.DataFrame(vals)
# List of column names
vals.columns = ['itemID_1', 'itemID_2', 'max_inter16', 'max_corr16', 'min_chisq16', 'min_hell16', 'max_hue10_inter', 'max_hue10_corr', 'min_hue10_inter', 'min_hue10_corr', 'xmed_inter16', 'xmed_corr16', 'xmed_chisq16', 'xmed_hell16', 'hue10_xmed_inter', 'hue10_xmed_corr', 'maxcor16_mean_max', 'maxcor16_mean_min', 'maxint16_mean_max', 'maxint16_mean_min', 'minhell16_mean_max', 'minhell16_mean_min', 'minchi16_mean_max', 'minchi16_mean_min', 'hue10_maxcor_mean_max', 'hue10_maxcor_mean_min', 'hue10_maxint_mean_max', 'hue10_maxint_mean_min', 'min_corr16', 'max_inter2', 'max_corr2', 'min_chisq2', 'min_hell2', 'max_hue150_inter', 'max_hue150_corr', 'min_hue150_inter', 'min_hue150_corr', 'xmed_inter2', 'xmed_corr2', 'xmed_chisq2', 'xmed_hell2', 'hue150_xmed_inter', 'hue150_xmed_corr', 'maxcor2_mean_max', 'maxcor2_mean_min', 'maxint2_mean_max', 'maxint2_mean_min', 'minhell2_mean_max', 'minhell2_mean_min', 'minchi2_mean_max', 'minchi2_mean_min', 'hue150_maxcor_mean_max', 'hue150_maxcor_mean_min', 'hue150_maxint_mean_max', 'hue150_maxint_mean_min', 'min_corr2']
# Add prefix to feature columns
vals.columns = ['set2c_' + c if c not in ['itemID_1', 'itemID_2'] else c for c in vals.columns]
a.print_elapsed(start)

start = time.time()
print('Caching data to disk ... ', end='', flush=True)
# Save updated dataset
if mode == 0:
    feather.write_dataframe(vals, cache_loc + 'features_train_set2c_histogram.fthr')
if mode == 1:
    feather.write_dataframe(vals, cache_loc + 'features_test_set2c_histogram.fthr')

a.print_elapsed(start)
print('set2c extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set2c_OK\n')
f.close()
