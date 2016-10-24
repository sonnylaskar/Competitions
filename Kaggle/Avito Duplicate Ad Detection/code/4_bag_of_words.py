#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Peter & Mikel
#### Avito Duplicate Ad Detection
# 4_bag_of_words.py
# Creates bags of word set intersection and differences, before running various linear models on them

import numpy as np
import pandas as pd
import sklearn
import json
from pandas.io.json import json_normalize
import unicodedata
import time
import codecs
import feather
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool

import libavito as a

# smoothening parameter for Naive Bayes
alpha = 1.0

# Options for Bag of words
ngram_range = (1, 1)
min_df = 20

print(a.c.BOLD + 'Extracting bag of words model features ...' + a.c.END)
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc

##### Preprocessing #####

start = time.time()
print('Reading input data ... ', end='', flush=True)
train = feather.read_dataframe(cache_loc + 'train.fthr')
test = feather.read_dataframe(cache_loc + 'test.fthr')
train = train[['itemID_1', 'itemID_2', 'cleantitle_1', 'cleantitle_2', 'cleandesc_1', 'cleandesc_2', 'isDuplicate']]
test = test[['itemID_1', 'itemID_2', 'cleantitle_1', 'cleantitle_2', 'cleandesc_1', 'cleandesc_2']]
a.print_elapsed(start)

train['testset'] = 0
test['testset'] = 1
train = pd.concat([train, test])
del test
gc.collect()

# BoW function
def preprocess_row(i):
    # Description
    dx = set(train.iloc[i]['cleandesc_1'].replace("'", "").split(" "))
    dy = set(train.iloc[i]['cleandesc_2'].replace("'", "").split(" "))
    dintersect = set.intersection(*[dx, dy])
    ddifference = set.symmetric_difference(*[dx, dy])

    # Title
    tx = set(train.iloc[i]['cleantitle_1'].replace("'", "").split(" "))
    ty = set(train.iloc[i]['cleantitle_2'].replace("'", "").split(" "))
    tintersect = set.intersection(*[tx, ty])
    tdifference = set.symmetric_difference(*[tx, ty])

    return [' '.join(dintersect), ' '.join(ddifference), ' '.join(tintersect), ' '.join(tdifference)]

# Create a list with co-occuring words per pair
desc_inter = []
desc_diff = []
title_inter = []
title_diff = []
t0 = time.time()
if nthreads == 1:
    print('Converting text to bag of words with 1 thread... ')
    for i in range(0, len(train.index)):
        if i % 10000 == 0:
            a.print_progress(i, t0, len(train.index))
        x = preprocess_row(i)
        desc_inter.append(x[0])
        desc_diff.append(x[1])
        title_inter.append(x[2])
        title_diff.append(x[3])
else:
    print('Converting text to bag of words multi-threaded ... ', end='', flush=True)
    pool = Pool(nthreads)
    ftrs = pool.map(preprocess_row, range(0, len(train.index)))
    pool.close()
    for x in ftrs:
        desc_inter.append(x[0])
        desc_diff.append(x[1])
        title_inter.append(x[2])
        title_diff.append(x[3])
    a.print_elapsed(t0)

train['title_same'] = pd.Series(title_inter, index=train.index)
train['title_diff'] = pd.Series(title_diff, index=train.index)
train['desc_same'] = pd.Series(desc_inter, index=train.index)
train['desc_diff'] = pd.Series(desc_diff, index=train.index)
train = train.drop(["cleantitle_1", "cleantitle_2", "cleandesc_1", "cleandesc_2"], 1)

models = ['title_same', 'title_diff', 'desc_same', 'desc_diff']

# Some vectors for convenience
trainidx = train.index[train['testset'] == 0]
testidx = train.index[train['testset'] == 1]
x = train['isDuplicate'].values

print("Parameters in this run:")
print("min_df   = ", min_df)
print("ngram    = ", ngram_range)

aucs = {}
for mod in models:
    print(a.c.BOLD + "Working on: " + mod + a.c.END)
    print('Creating vocabulary ... ', end='', flush=True)

    # create the vocabulary
    #---------------------
    # all words from test corpus
    t0 = time.time()
    docs = train[train['testset'] == 1][mod].tolist()
    count_vect = CountVectorizer(min_df=1, ngram_range=ngram_range)
    wordcounts_voc = count_vect.fit_transform(docs)
    vocabulary_test = count_vect.vocabulary_

    # all words from train corpus
    docs = train[train['testset'] == 0][mod].tolist()
    count_vect = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
    wordcounts_voc = count_vect.fit_transform(docs)
    vocabulary_train = count_vect.vocabulary_

    a.print_elapsed(t0)

    # Create a dictionary with all words from test having more than minf occurs in train
    vockey = set(vocabulary_train.keys()) & set(vocabulary_test.keys())
    print("Length of vocabulary train/test = ", len(vocabulary_train), len(vocabulary_test))
    vocabulary = {}
    ix = 0
    for i in vockey:
        vocabulary[i] = ix
        ix = ix + 1

    print("Length of vocabulary = ", len(vocabulary))

    print('Creating WOB features... ', end='', flush=True)
    # feature matrix for train
    t0 = time.time()
    docs = train[train['testset'] == 0][mod].tolist()
    count_vect = CountVectorizer(vocabulary=vocabulary)
    wordcounts_train = count_vect.fit_transform(docs)

    # feature matrix for all (used for prediction)
    docs = train[mod].tolist()
    count_vect = CountVectorizer(vocabulary=vocabulary)
    wordcounts_full = count_vect.fit_transform(docs)
    a.print_elapsed(t0)

    ########## CLASSIFIERS ##########

    t0 = time.time()
    print('Training models ...', end='', flush=True)

    # Train a Naive Bayes classifier
    #-------------------------------
    clf = BernoulliNB(binarize=0.5, alpha=alpha)
    clf.fit(wordcounts_train, x[trainidx])

    # Calculate AUCs
    predtrain = clf.predict_proba(wordcounts_train)
    predtrain = np.array(predtrain.T)[0]
    auc_train = roc_auc_score(x[trainidx], -predtrain)

    # save prediction for all
    predfull = clf.predict_proba(wordcounts_full)
    predfull = np.array(predfull.T)[0]
    train[mod + '_NB'] = pd.Series(predfull, index=train.index)
    aucs[mod + '_NB'] = auc_train

    # SGD classifier
    #------------------------
    for loss in ['log', 'modified_huber']:
        clf = SGDClassifier(loss=loss)
        clf.fit(wordcounts_train, x[trainidx])

        # Calculate AUCs
        predtrain = clf.predict_proba(wordcounts_train)
        predtrain = np.array(predtrain.T)[0]
        auc_train = roc_auc_score(x[trainidx], -predtrain)

        # save prediction for all
        predfull = clf.predict_proba(wordcounts_full)
        predfull = np.array(predfull.T)[0]
        train[mod + '_SGD_' + loss] = pd.Series(predfull, index=train.index)
        aucs[mod + '_SGD_' + loss] = auc_train

    for loss in ['huber', 'squared_loss']:
        clf = SGDRegressor(loss=loss)
        clf.fit(wordcounts_train, x[trainidx])

        # Calculate AUCs
        predtrain = clf.predict(wordcounts_train)
        auc_train = roc_auc_score(x[trainidx], predtrain)

        # save prediction for all
        predfull = clf.predict(wordcounts_full)
        train[mod + '_SGD_' + loss] = pd.Series(predfull, index=train.index)
        aucs[mod + '_SGD_' + loss] = auc_train

    a.print_elapsed(t0)

print('Models complete!')
print('train-AUCs: ' + str(aucs))

print('Caching data to disk ... ', end='', flush=True)
t0 = time.time()
test = train.loc[train.testset == 1]
train = train.loc[train.testset == 0]
train = train.drop(['testset', 'title_same', 'title_diff', 'desc_diff', 'desc_same'], 1)
test = test.drop(['isDuplicate', 'testset', 'title_same', 'title_diff', 'desc_diff', 'desc_same'], 1)

feather.write_dataframe(train, cache_loc + 'bow_models_train.fthr')
feather.write_dataframe(test, cache_loc + 'bow_models_test.fthr')

a.print_elapsed(t0)
print('Bag of words features complete!')
