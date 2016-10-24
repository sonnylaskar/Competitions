#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Mikel
#### Avito Duplicate Ad Detection
# 1_data_preprocessing.py
# Takes in input data, cleans text and merges itemIDs.

import numpy as np
import pandas as pd
import nltk
import sklearn
import json
import math
import feather  #import pickle - feather used instead as it is compatible with R
from pandas.io.json import json_normalize
import unicodedata
from stop_words import get_stop_words
import time
from multiprocessing import Pool
import sys
import gc
from imp import load_source

import libavito as a

#########################
##### SCRIPT CONFIG #####
#########################

# Define cleaning parameters
stopwords = get_stop_words('ru')
exclude_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po', 'Sk', 'Sc', 'So', 'Co', 'Cf', 'Cc', 'Cs', 'Cn'])
sno = nltk.stem.SnowballStemmer('russian')

#########################

print(a.c.BOLD + 'Cleaning input data ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '1_data_preprocessing.py')

## Read settings required by script
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc
category_loc = config.category_csv
location_loc = config.location_csv
debug = config.debug
if mode == 0:
    data_loc = config.train_ItemInfo
    pairs_loc = config.train_ItemPairs
if mode == 1:
    data_loc = config.test_ItemInfo
    pairs_loc = config.test_ItemPairs

# Read file for processing into memory
start = time.time()
print('Reading input data ... ', end='', flush=True)
df = pd.read_csv(data_loc)
a.print_elapsed(start)

def get_clean_tokens(text):
    newtext = []

    # lower text
    text = text.lower()

    # replace punctation
    text = ''.join(x if unicodedata.category(x) not in exclude_cats else ' ' for x in text)

    # replace some symbols
    text = ''.join(x if x not in ["'", '`', '>', '<', '=', '+'] else ' ' for x in text)

    # tokenize the text
    text0 = nltk.word_tokenize(text, 'russian')

    # word by word
    for y in text0:
        # remove stopwords and stemming
        if len(y) > 0 and y not in stopwords:
            newtext.append(sno.stem(y))

    return newtext

def process_line(i):
    # Lists to store tokens in
    tx = []
    dx = []
    resx = []

    # Pluck initial strings from dataframe
    title = str(df.iloc[i]['title'])
    desc = str(df.iloc[i]['description'])
    jx = str(df.iloc[i]['attrsJSON']).lower()

    tx = get_clean_tokens(title)
    dx = get_clean_tokens(desc)

    # Process JSON
    try:
        resx = json.loads(jx)
        for key in resx.keys():
            a = get_clean_tokens(resx[key])
            resx[key] = " ".join(a)
    except:
        resx = []
        if debug == 1:
            print('DEBUG: Failed to read JSON "' + json + '" at ' + str(i))
        pass

    jxs = '' + json.dumps(resx, ensure_ascii=False)
    txs = ' '.join(tx)
    dxs = ' '.join(dx)

    del tx, resx, dx
    gc.collect()

    return [txs, dxs, jxs]

# def process_line(i):
#     return ['empty', 'empty', 'empty']

newtitles = []
newdescs = []
newjson = []
ids = df['itemID'].values

start = time.time()
# If number of threads is equal to 1, output time remaining etc.
o = len(df.index)
if nthreads == 1:
    print('Cleaning text with 1 thread ...')
    k = 0
    # Iterate over lines
    for i in range(0, o):
        x = process_line(i)
        newtitles.append(x[0])
        newdescs.append(x[1])
        newjson.append(x[2])
        k += 1
        if k % 100 == 0:
            a.print_progress(k, start, o)
# Otherwise perform multi-threaded mapping
else:
    print('Cleaning text multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads)
    newdata = pool.map(process_line, range(0, o))
    pool.close()
    for x in newdata:
        newtitles.append(x[0])
        newdescs.append(x[1])
        newjson.append(x[2])

    del newdata
    gc.collect()

    a.print_elapsed(start)

#########################

print(a.c.BOLD + 'Joining input data ...' + a.c.END)

# Joining cleaned data into original data
df['cleandesc'] = newdescs
df['cleantitle'] = newtitles
df['cleanjson'] = newjson

# Memory management
del newdescs, newtitles, newjson
gc.collect()

start = time.time()
print('Joining parentCategory ... ', end='', flush=True)
category = pd.read_csv(category_loc)
df = df.merge(category, on=['categoryID'], copy=False)
a.print_elapsed(start)

start = time.time()
print('Joining regionID ... ', end='', flush=True)
location = pd.read_csv(location_loc)
df = df.merge(location, on=['locationID'], copy=False)
a.print_elapsed(start)

start = time.time()
print('Joining itemPairs ...', end='', flush=True)
itemPairs = pd.read_csv(pairs_loc)
df = pd.merge(pd.merge(itemPairs, df, how='inner', left_on='itemID_1', right_on='itemID'), df, how='inner', left_on='itemID_2', right_on='itemID')  # , suffixes=('_1', '_2'))
df.drop(['itemID_x', 'itemID_y'], axis=1, inplace=True)
df.columns = [c.replace('_x', '_1').replace('_y', '_2') for c in df.columns]
a.print_elapsed(start)

start = time.time()
print('Caching cleaned data ... ', end='', flush=True)

# Save updated dataset
if mode == 0:
    #pickle.dump(df, open(cache_loc + 'train.bin', 'wb'), protocol=4)
    feather.write_dataframe(df, cache_loc + 'train.fthr')
    df.to_csv(cache_loc + 'train.csv', index=False)
if mode == 1:
    #pickle.dump(df, open(cache_loc + 'test.bin', 'wb'), protocol=4)
    feather.write_dataframe(df, cache_loc + 'test.fthr')
    df.to_csv(cache_loc + 'test.csv', index=False)

a.print_elapsed(start)
print('Data preprocessing complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('data_preprocessing_OK\n')
f.close()
