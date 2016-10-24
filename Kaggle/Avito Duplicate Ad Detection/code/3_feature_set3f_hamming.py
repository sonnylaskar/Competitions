#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Peter & Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set3f_hamming.py
# Creates features from image dHashes

import pandas as pd
import numpy as np
import sys
import feather
import time
import gc
from multiprocessing import Pool

import libavito as a

def debug(s):
    print(str(s))
    time.sleep(1)

print(a.c.BOLD + 'Extracting set3f image hamming features ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set3f_hamming.py')

## Read settings required by script
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc
#debug = config.debug
if mode == 0:
    df = feather.read_dataframe(cache_loc + 'train.fthr')
if mode == 1:
    df = feather.read_dataframe(cache_loc + 'test.fthr')

root = config.images_root
image_db = feather.read_dataframe(cache_loc + 'image_database.fthr')

df = df[['itemID_1', 'itemID_2', 'images_array_1', 'images_array_2']]

start = time.time()
print('Preparing imageDB ... ', end='', flush=True)
image_db.index = image_db['image']
nhash = image_db['FreqOfHash'].to_dict()
ihash = image_db['imagehash'].to_dict()
a.print_elapsed(start)

def process_row(row):
    id1 = row[0]
    id2 = row[1]
    array_x = row[2]
    array_y = row[3]

    if array_x is not None:
        aux_x = array_x.replace(' ', '').split(',')
    else:
        aux_x = []
    if array_y is not None:
        aux_y = array_y.replace(' ', '').split(',')
    else:
        aux_y = []

    icount = []
    missing = 0
    minhamming = 99999
    minhamming30 = 99999
    minhamming50 = 99999
    minhamming100 = 99999
    #maxn = 0
    for k in range(0, 9):
        icount.append(0)

#   Find out if some images are repeated very often
    maxnx = 0
    maxny = 0
    for ix in aux_x:
        ix = int(ix)
        if ix in nhash:
            if maxnx < nhash[ix]:
                maxnx = nhash[ix]

    for iy in aux_y:
        iy = int(iy)
        if iy in nhash:
            if maxny < nhash[iy]:
                maxny = nhash[iy]

    for ix in aux_x:
        for iy in aux_y:
            if ix in ihash and iy in ihash:
                try:
                    a = int('0x' + ihash[ix], 16)
                    b = int('0x' + ihash[iy], 16)
                    hamming = bin(a ^ b).count("1")
                    if hamming < 9:
                        icount[hamming] = icount[hamming] + 1

                    if hamming < minhamming:
                        minhamming = hamming

                    if nhash[ix] < 100 and nhash[iy] < 100:
                        if minhamming100 > hamming:
                            minhamming100 = hamming

                    if nhash[ix] < 30 and nhash[iy] < 30:
                        if minhamming30 > hamming:
                            minhamming30 = hamming

                    if nhash[ix] < 50 and nhash[iy] < 50:
                        if minhamming50 > hamming:
                            minhamming50 = hamming

                except:
                    pass
                    #debug(['break', ix, iy])
            else:
                #debug(['missing', ix, iy])
                missing = missing + 1

    vals = [id1, id2] + icount + [missing, minhamming, maxnx, maxny, minhamming30, minhamming50, minhamming100]
    if min(len(aux_x), len(aux_y)) > 0:
        return vals
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

ftrs = []

start = time.time()
o = len(df.index)
if nthreads == 1:
    print('Extracting features with 1 thread ...')
    k = 0
    # Iterate over files
    ftrs = []
    for row in df.values:
        x = process_row(row)
        ftrs.append(x)
        k += 1
        if k % 100 == 0:
            a.print_progress(k, start, o)

# Otherwise perform multi-threaded mapping
else:
    print('Extracting features multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads)
    ftrs = pool.map(process_row, df.values)
    pool.close()
    gc.collect()

    a.print_elapsed(start)

ftrs = pd.DataFrame(ftrs)
ftrs = ftrs.loc[ftrs[0] > 0]
cols = ['itemID_1', 'itemID_2'] + [str(c) for c in ['ham' + str(i) for i in range(9)] + ['miss', 'minham', 'maxnx', 'maxny', 'minham30', 'minham50', 'minham100']]
print(cols)
ftrs.columns = cols

# Save updated dataset
if mode == 0:
    feather.write_dataframe(ftrs, cache_loc + 'features_train_set3f.fthr')
if mode == 1:
    feather.write_dataframe(ftrs, cache_loc + 'features_test_set3f.fthr')

a.print_elapsed(start)
print('set3f extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set3f_OK\n')
f.close()
