#!/bin/bash
################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : runAll.sh
# Description: This bash Script generates the Submission Files
# Usage:
#       bash ./runAll.sh
################################################################################################
################################################################################################


echo "`tput smso` Running Data Preprocessing`tput rmso`"
python3 code/1_data_preprocessing.py --train
python3 code/1_data_preprocessing.py --test

echo "`tput smso` Running Image Processing`tput rmso`"
python3 code/2_image_info.py

echo "`tput smso` Extracing NGrams`tput rmso`"
Rscript code/runnAll.sh train
Rscript code/runnAll.sh test

echo "`tput smso` Extracting NChars`tput rmso`"
Rscript code/3_feature_set1b_nchar.R train
Rscript code/3_feature_set1b_nchar.R test

echo "`tput smso` Extracting Misc Features`tput rmso`"
Rscript code/3_feature_set1c_misc.R train
Rscript code/3_feature_set1c_misc.R test

echo "`tput smso`Extracing Attributes `tput rmso`"
Rscript code/3_feature_set1e_attribute.R train
Rscript code/3_feature_set1e_attribute.R test

echo "`tput smso`Extracting Special Counting Features `tput rmso`"
Rscript code/3_feature_set1f_SpecialCounting.R train
Rscript code/3_feature_set1f_SpecialCounting.R test

echo "`tput smso` Extracting Capital Letters`tput rmso`"
Rscript code/3_feature_set1g_capitalLetters.R train
Rscript code/3_feature_set1g_capitalLetters.R test

echo "`tput smso` Extracting hash features `tput rmso`"
Rscript code/3_feature_set1h_images.R train
Rscript code/3_feature_set1h_images.R test

echo "`tput smso` Extracing Image Size Features `tput rmso`"
Rscript code/3_feature_set1i_imagesSize.R train
Rscript code/3_feature_set1i_imagesSize.R test

echo "`tput smso` Extracing Location `tput rmso`"
python3 code/3_feature_set2a_lev_loc.py --train
python3 code/3_feature_set2a_lev_loc.py --test

echo "`tput smso` Extracing BRISK`tput rmso`"
python3 code/3_feature_set2b_brisk.py --train
python3 code/3_feature_set2b_brisk.py --test

echo "`tput smso`Extracting Histograms `tput rmso`"
python3 code/3_feature_set2c_hist.py --train
python3 code/3_feature_set2c_hist.py --test

echo "`tput smso`Extracing Descriptions `tput rmso`"
python3 code/3_feature_set3a_description.py --train
python3 code/3_feature_set3a_description.py --test

echo "`tput smso`Extracting Title `tput rmso`"
python3 code/3_feature_set3b_title.py --train
python3 code/3_feature_set3b_title.py --test

echo "`tput smso` Extracting Json `tput rmso`"
python3 code/3_feature_set3c_json.py --train
python3 code/3_feature_set3c_json.py --test

echo "`tput smso` Extracing Jsonpart2 `tput rmso`"
python3 code/3_feature_set3d_json1.py --train
python3 code/3_feature_set3d_json1.py --test

echo "`tput smso`Extracing hamming `tput rmso`"
python3 code/3_feature_set3f_hamming.py --train
python3 code/3_feature_set3f_hamming.py --test

echo "`tput smso`Extracing Json to Col `tput rmso`"
python3 code/3_json_to_cols.py

echo "`tput smso`Extracing WOE `tput rmso`"
Rscript code/3_feature_set3g_json_to_cols_createWOE.R train
Rscript code/3_feature_set3g_json_to_cols_createWOE.R test

echo "`tput smso` Consolidating few features `tput rmso`"
Rscript code/3_feature_set3z_consolidate.R train
Rscript code/3_feature_set3z_consolidate.R test

echo "`tput smso` Extracing Fuzzy`tput rmso`"
python3 code/3_feature_set4a_fuzzy.py --train
python3 code/3_feature_set4a_fuzzy.py --test

echo "`tput smso` Extracting fuzzy Clean`tput rmso`"
python3 code/3_feature_set4b_fuzzy_clean.py --train
python3 code/3_feature_set4b_fuzzy_clean.py --test

echo "`tput smso`Extracing Alternate `tput rmso`"
python3 code/3_feature_set4c_alternate.py --train
python3 code/3_feature_set4c_alternate.py --test

echo "`tput smso` Extracing Similarity`tput rmso`"
python3 code/3_feature_set4d_similarity_clean.py --train
python3 code/3_feature_set4d_similarity_clean.py --test

echo "`tput smso`Extracing BOW `tput rmso`"
python3 code/4_bag_of_words.py



############################################################################################
############################################################################################
#Consolidate All Features
echo "`tput smso`CONSOLIDATING ALL FEATURES `tput rmso`"
Rscript code/5_consolidate_features.R train
Rscript code/5_consolidate_features.R test

echo "`tput smso`Replacing all NaN and Inf`tput rmso`"
python3 code/5_data_postprocessing.py --train
python3 code/5_data_postprocessing.py --test

echo "FEATURES DONE"
############################################################################################
echo "Running models"

echo "`tput smso`Running logit_v2`tput rmso`"
python2 code/models/marios_logit_v2.py

echo "`tput smso`Running nn_v1`tput rmso`"
python2 code/models/marios_nn_v1.py

echo "`tput smso`Running nnnew_v2`tput rmso`"
python2 code/models/marios_nnnew_v2.py

echo "`tput smso`Running nnnew_v3`tput rmso`"
python2 code/models/marios_nnnew_v3.py

echo "`tput smso`Running nnnew_v4`tput rmso`"
python2 code/models/marios_nnnew_v4.py

echo "`tput smso`Running ridge_v2`tput rmso`"
python2 code/models/marios_ridge_v2.py

echo "`tput smso`Running sgd_v2`tput rmso`"
python2 code/models/marios_sgd_v2.py

echo "`tput smso`Running xg_v1`tput rmso`"
python2 code/models/marios_xg_v1.py

echo "`tput smso`Running xgrank_v2`tput rmso`"
python2 code/models/marios_xgrank_v2.py

echo "`tput smso`Running xgrank_v3`tput rmso`"
python2 code/models/marios_xgrank_v3.py

echo "`tput smso`Running xgregv3`tput rmso`"
python2 code/models/marios_xgregv3.py

echo "`tput smso`Running xgson_v2`tput rmso`"
python2 code/models/marios_xgson_v2.py

echo "`tput smso`Running xgson_v3`tput rmso`"
python2 code/models/marios_xgson_v3.py

echo "`tput smso`Running xgson_v4`tput rmso`"
python2 code/models/marios_xgson_v4.py

echo "`tput smso`Running xgson_v2_v5`tput rmso`"
python2 code/models/marios_xgson_v2_v5.py

echo "`tput smso`Running meta-model`tput rmso`"
python2 code/models/meta_rf_v1.py

echo "MODELS DONE"
