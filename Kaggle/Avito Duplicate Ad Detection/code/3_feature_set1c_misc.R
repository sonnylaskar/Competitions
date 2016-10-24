################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1c_misc.R
# Description: This Rscript generates all ngram features
# Usage:
#       Rscript ./code/3_feature_set1c_misc.R train
#       Rscript ./code/3_feature_set1c_misc.R test
#       Default argument is test
################################################################################################
################################################################################################

args <- commandArgs(trailingOnly = F)
BASE <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))

# Source Config and functions.R file
source(paste(BASE, "/../config.cfg", sep = ""))
source(paste(BASE_DIR, "/code/functions.R", sep = ""))

#Load any additional packages
library(parallel)
library(stylo)

# Read argument for train or test
trainOrTest <- commandArgs(trailingOnly = TRUE)
if (length(trainOrTest) > 1) {
        stop("ERROR: I need only 1 argument : train or test")
}

if (length(trainOrTest) == 0) {
        print("No Arguments passed, Assuming you mean test")
        trainOrTest <- "test"
}

#Load data
FILENAME <- paste(cache_loc, "/", trainOrTest, ".csv", sep = "")
cat("Reading file ", FILENAME, "\n", sep = " ")
dat <- read_csv(FILENAME)


######## IDs and Long and Lat Features
print("Generating Binary features ")
isMetroIdSame <- ifelse(dat$metroID_1 == dat$metroID_2, 1, 0)
isLocationIDSame <- ifelse(dat$locationID_1 == dat$locationID_2, 1, 0)
isRegionIDSame <- ifelse(dat$regionID_1 == dat$regionID_2, 1, 0)
isLongitudeSame <-  ifelse(round(dat$lon_1, 2) == round(dat$lon_2, 2), 1, 0)
isLatitudeSame <-  ifelse(round(dat$lat_1, 2) == round(dat$lat_2, 2), 1, 0)
isTitleSame <- ifelse(tolower(dat$cleantitle_1) == tolower(dat$cleantitle_2), 1, 0) #isTitle Same
isdescriptionSame <- ifelse(tolower(dat$cleandesc_1) == tolower(dat$cleandesc_2), 1, 0) #isdescription Same

######## PRICE Features
print("Generating Price features ")
priceDiff <- abs(dat$price_1 - dat$price_2)
ratioOfPrices <- dat$price_1 / dat$price_2
ratioOfPrices <- round(ifelse(ratioOfPrices > 1, 1/ratioOfPrices, ratioOfPrices), 3)
both_price_na <- ifelse(is.na(dat$price_1) & is.na(dat$price_2), 1, 0) #Both Price NA
one_price_na <- ifelse(is.na(dat$price_1) | is.na(dat$price_2), 1, 0) #One Price NA
total_price <- (dat$price_1 + dat$price_2) #Total Price


######## IMAGE Features
print("Generating Image features")
library(stringr)
imageCount_sum <- str_count(dat$images_array_1, '[0-9.]+') + str_count(dat$images_array_2, '[0-9.]+')
imageCount_diff <- abs(str_count(dat$images_array_1, '[0-9.]+') - str_count(dat$images_array_2, '[0-9.]+'))
imageCount_min <- pmin(str_count(dat$images_array_1, '[0-9.]+'), str_count(dat$images_array_2, '[0-9.]+'),  na.rm = F)
imageCount_max <- pmax(str_count(dat$images_array_1, '[0-9.]+'), str_count(dat$images_array_2, '[0-9.]+'),  na.rm = F)
ratioOfNumberOfImages <- str_count(dat$images_array_1, '[0-9.]+') / str_count(dat$images_array_2, '[0-9.]+')
ratioOfNumberOfImages <- round(ifelse(ratioOfNumberOfImages > 1, 1/ratioOfNumberOfImages, ratioOfNumberOfImages), 3)

######## DISTANCE STRING Features
library(stringdist)
print("Generating Text Distance features for title")
titleDistance_cosine <- round(mcmapply(stringdist, dat$cleantitle_1, dat$cleantitle_2, method = "cosine", USE.NAMES = F, mc.cores = preprocessing_nthreads), 3)
titleDistance_hamming <- round(mcmapply(stringdist, dat$cleantitle_1, dat$cleantitle_2, method = "hamming", USE.NAMES = F, mc.cores = preprocessing_nthreads), 3)
titleDistance_jaccard <- round(mcmapply(stringdist, dat$cleantitle_1, dat$cleantitle_2, method = "jaccard", USE.NAMES = F, mc.cores = preprocessing_nthreads), 3)

print("Generating Text Distance features for description")
descriptionDistance_cosine <- round(mcmapply(stringdist, dat$cleandesc_1, dat$cleandesc_2, method = "cosine", USE.NAMES = F, mc.cores = preprocessing_nthreads), 3)

descriptionDistance_hamming <- round(mcmapply(stringdist, dat$cleandesc_1, dat$cleandesc_2, method = "hamming", USE.NAMES = F, mc.cores = preprocessing_nthreads), 3)

descriptionDistance_jaccard <- round(mcmapply(stringdist, dat$cleandesc_1, dat$cleandesc_2, method = "jaccard", USE.NAMES = F, mc.cores = preprocessing_nthreads), 3)


######## DATA FRAME
df_master <- data.frame(	isMetroIdSame = isMetroIdSame,
				isLocationIDSame = isLocationIDSame,
				isRegionIDSame = isRegionIDSame,
				isLongitudeSame = isLongitudeSame,
				isLatitudeSame = isLatitudeSame,
				isTitleSame = isTitleSame,
				isdescriptionSame = isdescriptionSame,
				priceDiff = priceDiff,
				ratioOfPrices = ratioOfPrices,
				both_price_na = both_price_na,
				one_price_na = one_price_na,
				total_price = total_price,
				imageCount_sum = imageCount_sum,
				imageCount_diff = imageCount_diff,
				imageCount_min = imageCount_min,
				imageCount_max = imageCount_max,
				ratioOfNumberOfImages = ratioOfNumberOfImages,
				titleDistance_cosine = titleDistance_cosine,
				titleDistance_hamming = titleDistance_hamming,
				titleDistance_jaccard = titleDistance_jaccard,
				descriptionDistance_cosine = descriptionDistance_cosine,
				descriptionDistance_hamming = descriptionDistance_hamming,
				descriptionDistance_jaccard = descriptionDistance_jaccard
			)

set1d <- df_master #making a copy for geenrating interaction features. Need to do this before renaming columns 

names(df_master) <- paste("set1c", names(df_master), sep = "_")
######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Misc features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1c_", "misc.fthr", sep = "" ))

# Start Interaction feature script
source("./code/3_feature_set1d_interaction.R")
#END
