#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Marios & Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set4d_similarity_clean.py
# Creates various similarity features on clean text, as well as the other columns

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

WORD = re.compile(r'\w+')

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def count_2words_together(words, text, ranges):
    count2 = 0
    if len(words) < 2 or len(text) < 2:
        return -1
    else:
        for m in range(0, len(words) - 1):
            words1 = words[m]
            for n in range(m + 1, len(words)):
                words2 = words[n]
                if words1 in text:
                    ind = text.index(words1)
                    try:
                        words2 in text[ind + 1:ind + 1 + ranges]
                        count2 += 1
                    except:
                        pass
        return count2

def count_2words(words, text):
        # To count how many times of the search terms having two words at least showing in texts.
    count2 = 0
    if len(words) < 2 or len(text) < 2:
        return -1
    else:
        for m in range(0, len(words) - 1):
            words1 = words[m]
            for n in range(m + 1, len(words)):
                words2 = words[n]
                if words1 in text and words2 in text:
                    count2 += 1
        return count2

def calculate_similarity_simple(str1, str2):
    count = 0
    if str1 in str2:
        count = 1
    return count

def calculate_similarity_split(str1, str2):
    count = 0
    countabs = 0
    countper = 0
    split1 = str1.split(" ")
    split2 = str2.split(" ")
    for s1 in split1:
        for s2 in split2:
            if s1 in s2:
                count += 1
            if s1 == s2:
                countabs += 1
            countper += 1

    return count, countabs, countabs / (countper + 1)

def get_string_value(string_value):
    if string_value in ["NA", np.nan]:
        return -1
    else:
        return float(string_value)

def process_row(row):

    title = 2
    desc = 4
    json = 6
    imagearray = 8
    lat = 10
    lon = 12
    price = 14
    catid = 16
    locid = 18
    metid = 20
    pcatid = 22
    regid = 24

    values = []

    values.append(row[0])
    values.append(row[1])

    # Magic to gracefully handle NaNs
    row = [i if i is not None else "NA" for i in row]

    # string feature counts
    for d in [title, desc, json]:
        st_1 = row[d]
        st_2 = row[d + 1]
        values.append(float(len(st_1) + 1.0))
        values.append(float(len(st_1) + 1.0) / float(len(st_2) + 1.0))
        values.append(float(len(st_1.split(" ")) + 1.0))
        values.append(float(len(st_1.split(" ")) + 1.0) / float(len(st_2.split(" ")) + 1.0))
        values.append(calculate_similarity_simple(st_1, st_2))
        val1, val2, val3 = calculate_similarity_split(st_1, st_2)
        values.append(val1)
        values.append(val2)
        values.append(val3)
        st_1_array = st_1.split(" ")
        st_2_array = st_2.split(" ")
        values.append(count_2words(st_1_array, st_2_array))
        values.append(get_cosine(st_1, st_2))
        values.append(count_2words_together(st_1_array, st_2_array, 1))
        values.append(count_2words_together(st_1_array, st_2_array, 5))

    st_1 = row[imagearray]
    st_2 = row[imagearray + 1]

    values.append(float(len(st_1.split(" ")) + 1.0))
    values.append(float(len(st_1.split(" ")) + 1.0) / float(len(st_2.split(" ")) + 1.0))
    # numerical feature values
    for d in [lat, lon, price]:
        values.append(get_string_value(row[d]))
        values.append(get_string_value(row[d]) / get_string_value(row[d + 1]))
    counters = 0.0
    for d in [catid, locid, metid, pcatid, regid]:
        if row[d] == row[d + 1] and row[d] not in ["NA", np.nan]:
            values.append(1.0)
            counters += 1.0
        else:
            values.append(0.0)
    values.append(counters / 5.0)

    return values

print(a.c.BOLD + 'Extracting set4d clean similarity features ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set4d_similarity_clean.py')

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

df = df[['itemID_1', 'itemID_2', 'cleantitle_1', 'cleantitle_2', 'cleandesc_1', 'cleandesc_2', 'cleanjson_1', 'cleanjson_2', 'images_array_1', 'images_array_2', 'lat_1', 'lat_2', 'lon_1', 'lon_2', 'price_1', 'price_2', 'categoryID_1', 'categoryID_2', 'locationID_1', 'locationID_2', 'metroID_1', 'metroID_2', 'parentCategoryID_1', 'parentCategoryID_2', 'regionID_1', 'regionID_2']]

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
cols = ['itemID_1', 'itemID_2'] + ['set4d_X' + str(i) for i in range(1, len(ftrs.columns.tolist()) - 1)]
print(cols)
ftrs.columns = cols

# Save updated dataset
if mode == 0:
    feather.write_dataframe(ftrs, cache_loc + 'features_train_set4d_similarity.fthr')
if mode == 1:
    feather.write_dataframe(ftrs, cache_loc + 'features_test_set4d_similarity.fthr')

a.print_elapsed(start)
print('set4d extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set4d_OK\n')
f.close()
