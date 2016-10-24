#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Marios & Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set4b_fuzzy_clean.py
# Creates various text similarity features

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

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


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

def process_row(row):

    title = 2
    desc = 4
    json = 6

    pairs = [[title, desc], [desc, title], [title, json], [json, title], [desc, json], [json, desc]]
    values = []
    # string feature counts

    values.append(row[0])
    values.append(row[1])

    for d, s in pairs:
        st_1 = str(row[d]).replace(":", " ")
        st_2 = str(row[s + 1]).replace(":", " ")
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

    return values

print(a.c.BOLD + 'Extracting set4c alternate text features ...' + a.c.END)

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set4c_fuzzy_clean.py')

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
cols = ['itemID_1', 'itemID_2'] + ['set4c_X' + str(i) for i in range(1, len(ftrs.columns.tolist()) - 1)]
print(cols)
ftrs.columns = cols

# Save updated dataset
if mode == 0:
    feather.write_dataframe(ftrs, cache_loc + 'features_train_set4c_alternate.fthr')
if mode == 1:
    feather.write_dataframe(ftrs, cache_loc + 'features_test_set4c_alternate.fthr')

a.print_elapsed(start)
print('set4c extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set4c_OK\n')
f.close()
