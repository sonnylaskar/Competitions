#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Marios & Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set4e_count3way_clean.py
# Counts how many 3-random-grams in item1 appear in item2

import numpy as np
import pandas as pd
import sys
import jellyfish
import feather
import time
import gc
import re
import math
from collections import Counter
from fuzzywuzzy import fuzz
from multiprocessing import Pool

import libavito as a

def count_3words(words, text):
    # To count how many times of the search terms having three words at least showing in texts.
    count3 = 0
    if len(words) < 3 or len(text) < 3:
        return -1
    else:
        for m in range(0, len(words) - 2):
            words1 = words[m]
            for n in range(m + 1, len(words) - 1):
                words2 = words[n]
                for z in range(m + 2, len(words)):
                    words3 = words[z]
                    if words1 in text and words2 and words3 in text:
                        count3 += 1
        return count3

def process_row(row):

    title = 2
    desc = 4
    json = 6

    values = []

    values.append(row[0])
    values.append(row[1])

    for d in [title, desc, json]:
        st_1 = str(row[d]).replace(":", " ").replace('"', ' ')
        st_2 = str(row[d + 1]).replace(":", " ").replace('"', ' ')
        values.append(count_3words(st_1.split(" "), st_2.split(" ")))

    return values

print(a.c.BOLD + 'Extracting set4e 3-way word count features ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set4e_fuzzy_clean.py')

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
    df = feather.read_dataframe(cache_loc + 'test.fthr')[:1000]

df = df[['itemID_1', 'itemID_2', 'cleantitle_1', 'cleantitle_2', 'cleandesc_1', 'cleandesc_2', 'cleanjson_1', 'cleanjson_2']]

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
        if k % 1 == 0:
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
cols = ['itemID_1', 'itemID_2'] + ['set4e_X' + str(i) for i in range(1, len(ftrs.columns.tolist()) - 1)]
print(cols)
ftrs.columns = cols

# Save updated dataset
if mode == 0:
    feather.write_dataframe(ftrs, cache_loc + 'feature_set4e_train.fthr')
if mode == 1:
    feather.write_dataframe(ftrs, cache_loc + 'feature_set4e_test.fthr')

a.print_elapsed(start)
print('set4e extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set4e_OK\n')
f.close()
