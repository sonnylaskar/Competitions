#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Marios & Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set4a_fuzzy.py
# Creates text features using the fuzzywuzzy python packages

import numpy as np
import pandas as pd
import sys
import time
import gc
import feather
from fuzzywuzzy import fuzz
from multiprocessing import Pool

import libavito as a

def process_row(row):
    values = []
    values.append(row[0])
    values.append(row[1])

    # Not black magic, iterate over title/description/json
    for d in [2, 4, 6]:
        st_1 = str(row[d])
        st_2 = str(row[d + 1])
        values.append(fuzz.partial_ratio(st_1, st_2))
        values.append(fuzz.token_set_ratio(st_1, st_2))
        values.append(fuzz.ratio(st_1, st_2))
        values.append(fuzz.token_sort_ratio(st_1, st_2))
    return values

print(a.c.BOLD + 'Extracting set4a fuzzy text features ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set4a_fuzzy.py')

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

df = df[['itemID_1', 'itemID_2', 'title_1', 'title_2', 'description_1', 'description_2', 'attrsJSON_1', 'attrsJSON_2']]

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
cols = ['itemID_1', 'itemID_2'] + ['set4a_X' + str(i) for i in range(1, len(ftrs.columns.tolist()) - 1)]
print(cols)
ftrs.columns = cols

# Save updated dataset
if mode == 0:
    feather.write_dataframe(ftrs, cache_loc + 'features_train_set4a_fuzzy.fthr')
if mode == 1:
    feather.write_dataframe(ftrs, cache_loc + 'features_test_set4a_fuzzy.fthr')

a.print_elapsed(start)
print('set4a extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set4a_OK\n')
f.close()
