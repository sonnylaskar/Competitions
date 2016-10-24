# Kaggle Avito Duplicate Ad Detection Contest 
Winning Solution Blog : https://blog.kaggle.com/2016/08/31/avito-duplicate-ads-detection-winners-interview-2nd-place-team-the-quants-mikel-peter-marios-sonny/

Contest Link: https://www.kaggle.com/c/avito-duplicate-ads-detection/

Private Leaderboard Score - _0.95294_ ( Rank 2 / 548)

Final solution of Avito Duplicate Ad Detection - TheQuants

##Prerequisites:
**OS:** Any Linux Distribution (Ubuntu 14.04 Preferred)  
**RAM:** 128GB+ (64GB for feature extraction)  
**CPU:** 36 cores+ (Preferred)  
**GPU:** CUDA-compatible NVIDIA GPU with Compute Capability 3.5+ (TITAN X Preferred)  
**Storage:** 64GB+ (not including input data) - Images on SSD _highly recommended_

**R Version:** 3.1+  
**R Packages:** data.table, dplyr, dummies, feather, Hmisc, igraph, jsonlite, parallel, raster, readr, reshape2, stringdist, stringr, stylo, textreuse, tidyr, tm, xgboost

**Python Version:** 3.5.1  
**Python Packages:** scikit-learn, numpy, pandas, python-Levenshtein, codecs, OpenCV, feather-format, jellyfish, nltk, PIL, fuzzywuzzy, stop_words, haversine

**Python Version:** 2.7.1  
**Python Packages:**  scikit-learn, feather-format, numpy, pandas
XGBoost (0.4.0)  
Keras (0.3.2)  
Theano (0.8.0rc1)  

##How to Generate the Submission File  
1) Update `config.cfg` and set all config parameters  
2) Ensure all directories mentioned in config.cfg are write-able  
3) Run `RunAll.sh`  

_Note_: In order to generate the full submission including models, it may take several weeks and needs at least 128GB of RAM
