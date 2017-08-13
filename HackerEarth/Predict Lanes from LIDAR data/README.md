# Approach 
# Sonny Laskar https://github.com/sonnylaskar

## Pre-requisites:
R 3.3+
Packages: xgboost, tidyverse, feather, geosphere

## Approach

### Directory
Create folders - code, input, output
Copy all input files in input folder
Copy all code files in code folder

### Build Haversine Length of Each Line
Execute Rscript final_1_calculateHaversineDistance.R

Execute Rscript final_2_buildData.R
        This script builds all features and prepares the data for final model

Execute final_3_buildModel.R to build the final model
        XGBOOST models with 10 different seeds are built and averaged
        The final submission file will be in output folder
# Approach

## Pre-requisites:
R 3.3+
Packages: xgboost, tidyverse, feather, geosphere, stringr

## Approach

### Directory
Create folders - code, input, output
Copy all input files in input folder
Copy all code files in code folder

### Build Haversine Length of Each Line
Execute Rscript final_1_calculateHaversineDistance.R

Execute Rscript final_2_buildData.R
        This script builds all features and prepares the data for final model

Feature Engineering:
  sumOfDistanceFromLeft = Sum of all distances towards Left
  sumOfDistanceFromRight = Sum of all distances towards Right
  r_sumOfDistanceFromLR  = Rati of the above two
  int_distLR = Intersection between the distances in left and right
  latCounter = Unique Count of latitude after rounding off to 4 digits
  lonCounter = Unique Count of longitude after rounding off to 4 digits
  uniq_linesLeft = Unique lines on Left
  uniq_linesRight = Unique lines on Right
  totalLaneLinesMean = Mean of total Lane Lines
  haversineDistance = Haversine length of each line and averaged, Then it is scaled by dividing against the LaneLineMean value
  [Refer feature Importance plot for importance]

Execute final_3_buildModel.R to build the final model
        XGBOOST models with 10 different seeds are built and averaged
        The final submission file will be in output folder
