#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar, Peter Borrmann & Marios Michailidis // TheQuants
#### Author: Mikel
#### Avito Duplicate Ad Detection
# 3_feature_set2_lev_loc.py
# This script takes in cleaned data and computes levenshtein distances as well as fuzzy distance from location clusters

import pandas as pd
import numpy as np
import feather
import time
import gc
import random
import sys
import Levenshtein  # pip install python-Levenshtein
from haversine import haversine

import libavito as a

# Noise to add to variables to prevent overfitting, a value between +- the selected value will be added to every instance
tot_lon_noise = 0.25
tot_lat_noise = 0.25
loc_dist_noise = 10

# Get train/test mode from launch argument
mode = a.get_mode(sys.argv, '3_feature_set2a_lev_loc.py')

## Read settings required by script
config = a.read_config()
nthreads = config.preprocessing_nthreads
cache_loc = config.cache_loc
debug = config.debug
if mode == 0:
    df = feather.read_dataframe(cache_loc + 'train.fthr')
if mode == 1:
    df = feather.read_dataframe(cache_loc + 'test.fthr')

# Create dataframe for features
x_all = pd.DataFrame()

random.seed(2016)
np.random.seed(2016)

########## FEATURE EXTRACTION ##########

print(a.c.BOLD + 'Extracting set2a features ...' + a.c.END)

## Location features
print(a.c.BOLD + 'Finding location features ...' + a.c.END)

start = time.time()
print('Finding total longtitude & adding noise of ' + str(tot_lon_noise) + ' ... ', end='', flush=True)
#x_all['tot_lon'] = df['lon_x'] + df['lon_y']
vals = []
data = df[['lon_1', 'lon_2']].values.tolist()
for x in data:
    vals.append(int(x[0]) + int(x[1]) + random.uniform(-tot_lon_noise, tot_lon_noise))
x_all['tot_lon'] = vals
a.print_elapsed(start)

start = time.time()
print('Finding total latitude & adding noise of ' + str(tot_lat_noise) + '... ', end='', flush=True)
#x_all['tot_lat'] = df['lat_x'] + df['lat_y']
vals = []
data = df[['lat_1', 'lat_2']].values.tolist()
for x in data:
    vals.append(int(x[0]) + int(x[1]) + random.uniform(-tot_lon_noise, tot_lon_noise))
x_all['tot_lat'] = vals
a.print_elapsed(start)

start = time.time()
print('Finding distance from location keypoints & adding noise of ' + str(loc_dist_noise) + '... ', end='', flush=True)
kalingrad_dist = []
moscow_dist = []
petersburg_dist = []
krasnodar_dist = []
makhachkala_dist = []
murmansk_dist = []
perm_dist = []
omsk_dist = []
khabarovsk_dist = []
klyuchi_dist = []
norilsk_dist = []
kalingrad = (20, 54)
moscow = (37, 55)
petersburg = (30, 60)
krasnodar = (39, 45)
makhachkala = (48, 43)
murmansk = (33, 69)
perm = (56, 58)
omsk = (73, 55)
khabarovsk = (135, 48.5)
klyuchi = (160, 56)
norilsk = (70, 90)
data = df[['lon_1', 'lat_1', 'lon_2', 'lat_2']].values.tolist()
for d in data:
    x = (d[1], d[0])
    y = (d[3], d[2])
    kalingrad_dist.append(haversine(kalingrad, x) + haversine(kalingrad, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    moscow_dist.append(haversine(moscow, x) + haversine(moscow, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    petersburg_dist.append(haversine(petersburg, x) + haversine(petersburg, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    krasnodar_dist.append(haversine(krasnodar, x) + haversine(krasnodar, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    makhachkala_dist.append(haversine(makhachkala, x) + haversine(makhachkala, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    murmansk_dist.append(haversine(murmansk, x) + haversine(murmansk, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    perm_dist.append(haversine(perm, x) + haversine(perm, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    omsk_dist.append(haversine(omsk, x) + haversine(omsk, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    khabarovsk_dist.append(haversine(khabarovsk, x) + haversine(khabarovsk, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    klyuchi_dist.append(haversine(klyuchi, x) + haversine(klyuchi, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
    norilsk_dist.append(haversine(norilsk, x) + haversine(norilsk, y) + random.uniform(-loc_dist_noise, loc_dist_noise))
x_all['kalingrad_dist'] = kalingrad_dist
x_all['moscow_dist'] = moscow_dist
x_all['petersburg_dist'] = petersburg_dist
x_all['krasnodar_dist'] = krasnodar_dist
x_all['makhachkala_dist'] = makhachkala_dist
x_all['murmansk_dist'] = murmansk_dist
x_all['perm_dist'] = perm_dist
x_all['omsk_dist'] = omsk_dist
x_all['khabarovsk_dist'] = khabarovsk_dist
x_all['klyuchi_dist'] = klyuchi_dist
x_all['norilsk_dist'] = norilsk_dist
a.print_elapsed(start)

## Levenshtein features
print(a.c.BOLD + 'Finding levenshtein features ...' + a.c.END)

start = time.time()
print('Finding Levenshtein distance between titles ... ', end='', flush=True)
vals = []
data = df[['title_1', 'title_2']].values.tolist()
for x in data:
    #vals.append(levenshtein(str(x[0]), str(x[1])))
    vals.append(Levenshtein.distance(str(x[0]), str(x[1])))
x_all['title_lev'] = vals
a.print_elapsed(start)

start = time.time()
print('Finding total title length ... ', end='', flush=True)
vals = []
data = df[['title_1', 'title_2']].values.tolist()
for x in data:
    vals.append(len(str(x[0])) + len(str(x[1])))
title_tot_len = vals
a.print_elapsed(start)

start = time.time()
print('Finding normalised Levenshtein distance ...', end='', flush=True)
x_all['title_lev_norm'] = x_all['title_lev'] / title_tot_len
a.print_elapsed(start)

start = time.time()
print('Finding Levenshtein distance between stemmed titles ... ', end='', flush=True)
vals = []
data = df[['cleantitle_1', 'cleantitle_2']].values.tolist()
for x in data:
    #vals.append(levenshtein(str(x[0]), str(x[1])))
    vals.append(Levenshtein.distance(str(x[0]), str(x[1])))
x_all['stem_title_lev'] = vals
a.print_elapsed(start)

start = time.time()
print('Finding total stemmed title length ... ', end='', flush=True)
vals = []
data = df[['title_1', 'title_2']].values.tolist()
for x in data:
    vals.append(len(str(x[0])) + len(str(x[1])))
stem_title_tot_len = vals
a.print_elapsed(start)

start = time.time()
print('Finding normalised Levenshtein distance ...', end='', flush=True)
x_all['stem_title_lev_norm'] = x_all['stem_title_lev'] / stem_title_tot_len
a.print_elapsed(start)

# Add set2a prefix to features
x_all.columns = ['set2a_' + c for c in x_all.columns]

# Keep itemIDs
x_all['itemID_1'] = df['itemID_1']
x_all['itemID_2'] = df['itemID_2']

start = time.time()
print('Caching data to disk ... ', end='', flush=True)
# Save updated dataset
if mode == 0:
    feather.write_dataframe(x_all, cache_loc + 'features_train_set2a_location_levenshtein.fthr')
if mode == 1:
    feather.write_dataframe(x_all, cache_loc + 'features_test_set2a_location_levenshtein.fthr')

a.print_elapsed(start)
print('set2a extraction complete!')

# Write status to status file so master script knows whether to proceed.
f = open(cache_loc + 'status.txt', 'a')
f.write('feature_set2a_OK\n')
f.close()
