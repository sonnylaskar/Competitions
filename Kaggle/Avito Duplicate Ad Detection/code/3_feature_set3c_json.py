#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Peter & Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set3c_json.py
# Creates some features from clean jsons

import numpy as np
import pandas as pd
import nltk
import sklearn
import json
import sys
import gc
import feather
from pandas.io.json import json_normalize
import unicodedata
from stop_words import get_stop_words
import time
from multiprocessing import Pool

import libavito as a

stopwords = get_stop_words('ru')
punctutation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
sno = nltk.stem.SnowballStemmer('russian')

def get_clean_tokens(text):
    newtext = []
    text0 = nltk.word_tokenize(text, 'russian')
    for y in text0:
        y = ''.join(x for x in y
                  if unicodedata.category(x) not in punctutation_cats)
        if len(y) > 0 and y not in stopwords:
            newtext.append(sno.stem(y))
    return newtext

def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    if union_cardinality == 0:
        return -1.0
    else:
        return intersection_cardinality / float(union_cardinality)

def ratio_of_matches(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    x_cardinality = len(x)
    if x_cardinality == 0:
        return -1.0
    else:
        return intersection_cardinality / float(x_cardinality)

print(a.c.BOLD + 'Extracting set3c JSON features ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set3c_json.py')

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

train = df[['itemID_1', 'itemID_2', 'attrsJSON_1', 'attrsJSON_2']]
del df
gc.collect()

train = train.fillna('')

ftrs = []

def process_row(i):
    jx = get_clean_tokens(train.iloc[i]['attrsJSON_1'])
    jy = get_clean_tokens(train.iloc[i]['attrsJSON_2'])
    sim_j = jaccard_similarity(jx, jy)
    mat1_j = ratio_of_matches(jx, jy)
    mat2_j = ratio_of_matches(jy, jx)
    return [train.iloc[i]['itemID_1'], train.iloc[i]['itemID_2'], sim_j, mat1_j, mat2_j]

t0 = time.time()
if nthreads == 1:
    print('Extracting features with 1 thread ...')
    for i in range(0, len(train.index)):
        if i % 10000 == 0:
            a.print_progress(i, t0, len(train.index))
        ftrs.append(process_row(i))
else:
    print('Extracting features multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads)
    ftrs = pool.map(process_row, range(0, len(train.index)))
    pool.close()
    a.print_elapsed(t0)

start = time.time()
print('Caching data to disk ... ', end='', flush=True)
ftrs = pd.DataFrame(ftrs)
ftrs.columns = ['itemID_1', 'itemID_2', 'simjson', 'matjson1', 'matjson2']

# Save updated dataset
if mode == 0:
    feather.write_dataframe(ftrs, cache_loc + 'features_train_set3c.fthr')
if mode == 1:
    feather.write_dataframe(ftrs, cache_loc + 'features_test_set3c.fthr')

a.print_elapsed(start)
print('set3c extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set3c_OK\n')
f.close()
