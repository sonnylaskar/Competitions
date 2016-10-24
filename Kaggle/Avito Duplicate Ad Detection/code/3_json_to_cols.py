#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Peter & Mikel
#### Avito Duplicate Ad Detection
# 3_json_to_cols.py
# Encodes json key similarity into a sparse format for feature extraction

import numpy as np
import pandas as pd
import sklearn
import json
from pandas.io.json import json_normalize
import unicodedata
import time
import codecs
import feather

import libavito as a

def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    if union_cardinality == 0:
        return -1.0
    else:
        return intersection_cardinality / float(union_cardinality)

## Read settings required by script
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc
debug = config.debug
df_train = feather.read_dataframe(cache_loc + 'train.fthr')
df_test = feather.read_dataframe(cache_loc + 'test.fthr')

df_train = df_train[['itemID_1', 'itemID_2', 'cleanjson_1', 'cleanjson_2']]
df_test = df_test[['itemID_1', 'itemID_2', 'cleanjson_1', 'cleanjson_2']]

df = pd.concat([df_train, df_test])

clean_jsons = df['cleanjson_1'].tolist() + df['cleanjson_2'].tolist()

print('Creating key dict ... ')
allkey = {}
pa = 0
t0 = time.time()
for i in range(0, len(clean_jsons)):
    if i % 100000 == 0:
        a.print_progress(i, t0, len(clean_jsons))
    try:
        jx = clean_jsons[i].replace("'", "")
        resx = json.loads(jx)
        for x in resx.keys():
            if x in allkey:
                allkey[x] = allkey[x] + 1
            else:
                allkey[x] = 1
    except KeyboardInterrupt:
        raise
    except Exception as e:
        pa += 1

t0 = time.time()
print('Transforming key dict ... ', end='', flush=True)
icount = 0
keydict = {}
for k, n in allkey.items():
    keydict[k] = icount
    icount += 1
a.print_elapsed(t0)

ftrs_train = []
print('Generating for train ... ')
t0 = time.time()
pa = 0
for i in range(0, len(df_train.index)):
    if i % 10000 == 0:
        a.print_progress(i, t0, len(df_train.index))
    try:
        jx = df_train.iloc[i]['cleanjson_1'].replace("'", "")
        jy = df_train.iloc[i]['cleanjson_2'].replace("'", "")
        resx = json.loads(jx)
        resy = json.loads(jy)
    except KeyboardInterrupt:
        raise
    except:
        continue

    if resx != [] and resy != []:
        for key in set.union(*[set(resx.keys()), set(resy.keys())]):
            if key in resx.keys() and key in resy.keys():
                c = resx[key]
                b = resy[key]
                res = jaccard_similarity(c, b)
            else:
                res = -1
            ftrs_train.append([df_train.iloc[i]['itemID_1'], df_train.iloc[i]['itemID_2'], str(keydict[key]), str(res)])
    else:
        pa += 1

ftrs_test = []
print('Generating for test ... ')
t0 = time.time()
for i in range(0, len(df_test.index)):
    if i % 10000 == 0:
        a.print_progress(i, t0, len(df_test.index))
    try:
        jx = df_test.iloc[i]['cleanjson_1'].replace("'", '')
        jy = df_test.iloc[i]['cleanjson_2'].replace("'", '')
        resx = json.loads(jx)
        resy = json.loads(jy)
    except KeyboardInterrupt:
        raise
    except:
        continue

    if resx != [] and resy != []:
        for key in set.union(*[set(resx.keys()), set(resy.keys())]):
            if key in resx.keys() and key in resy.keys():
                c = resx[key]
                b = resy[key]
                res = jaccard_similarity(c, b)
            else:
                res = -1
            ftrs_test.append([df_test.iloc[i]['itemID_1'], df_test.iloc[i]['itemID_2'], str(keydict[key]), str(res)])
    else:
        pa += 1

print("\nError rows: " + str(pa))

print(len(ftrs_train))
print(len(ftrs_test))

print('Tranforming features ... ', end='', flush=True)
t0 = time.time()
ftrs_train = pd.DataFrame(ftrs_train)
ftrs_test = pd.DataFrame(ftrs_test)
ftrs_train.columns = ['itemID_1', 'itemID_2', 'keyID', 'value']
ftrs_test.columns = ['itemID_1', 'itemID_2', 'keyID', 'value']
a.print_elapsed(t0)

print('Caching data to disk ... ', end='', flush=True)
t0 = time.time()
feather.write_dataframe(ftrs_train, cache_loc + 'json_vals_train_v2.fthr')
feather.write_dataframe(ftrs_test, cache_loc + 'json_vals_test_v2.fthr')
a.print_elapsed(t0)

print('json_to_cols Complete!')
