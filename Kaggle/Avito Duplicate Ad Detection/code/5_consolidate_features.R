################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1a_ngram.R
# Description: This Rscript generates all ngram features
# Usage:
#       Rscript ./code/3_feature_set1a_ngram.R train
#       Rscript ./code/3_feature_set1a_ngram.R test
#       Default argument is test
################################################################################################
################################################################################################

# Source Config and functions.R file
source("config.cfg")
source("./code/functions.R")

library(readr)
library(dplyr)
library(feather)


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
completeDate <- read_csv(FILENAME)
if (trainOrTest == "train") {
	completeDate <- completeDate[, c("itemID_1", "itemID_2", "isDuplicate")]
	gc()
} else {
	completeDate <- completeDate[, c("id", "itemID_1", "itemID_2")]
	gc()
}     

ngram_title <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1a_ngram_title.fthr", sep = "" ))
completeDate <- left_join(completeDate, ngram_title, by = c("itemID_1", "itemID_2"))
rm(ngram_title)

ngram_description <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1a_ngram_description.fthr", sep = "" ))
completeDate <- left_join(completeDate, ngram_description, by = c("itemID_1", "itemID_2"))
rm(ngram_description)

nchar_title <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1b_nchar_title.fthr", sep = "" ))
completeDate <- left_join(completeDate, nchar_title, by = c("itemID_1", "itemID_2"))
rm(nchar_title)

nchar_description <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1b_nchar_description.fthr", sep = "" ))
completeDate <- left_join(completeDate, nchar_description, by = c("itemID_1", "itemID_2"))
rm(nchar_description)

misc <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1c_misc.fthr", sep = "" ))
completeDate <- left_join(completeDate, misc, by = c("itemID_1", "itemID_2"))
rm(misc)

interaction <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1d_interaction.fthr", sep = "" ))
completeDate <- left_join(completeDate, interaction, by = c("itemID_1", "itemID_2"))
rm(interaction)

attributes <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1e_attributes.fthr", sep = "" ))
completeDate <- left_join(completeDate, attributes, by = c("itemID_1", "itemID_2"))
rm(attributes)

specialCounting <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1f_specialCounting.fthr", sep = "" ))
completeDate <- left_join(completeDate, specialCounting, by = c("itemID_1", "itemID_2"))
rm(specialCounting)

capitalLetters <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1g_capitalLetters.fthr", sep = "" ))
completeDate <- left_join(completeDate, capitalLetters, by = c("itemID_1", "itemID_2"))
rm(capitalLetters)

image <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1h_image.fthr", sep = "" ))
completeDate <- left_join(completeDate, image, by = c("itemID_1", "itemID_2"))
rm(image)

imageSize <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set1i_imageSize.fthr", sep = "" ))
completeDate <- left_join(completeDate, imageSize, by = c("itemID_1", "itemID_2"))
rm(imageSize)



location_levenshtein <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set2a_location_levenshtein.fthr", sep = "" ))
completeDate <- left_join(completeDate, location_levenshtein, by = c("itemID_1", "itemID_2"))
rm(location_levenshtein)

brisk <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set2b_brisk.fthr", sep = "" ))
completeDate <- left_join(completeDate, brisk, by = c("itemID_1", "itemID_2"))
rm(brisk)

histogram <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set2c_histogram.fthr", sep = "" ))
completeDate <- left_join(completeDate, histogram, by = c("itemID_1", "itemID_2"))
rm(histogram)


consolidated <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set3_consolidated.fthr", sep = "" ))
completeDate <- left_join(completeDate, consolidated, by = c("itemID_1", "itemID_2"))
rm(consolidated)


fuzzy <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set4a_fuzzy.fthr", sep = "" ))
completeDate <- left_join(completeDate, fuzzy, by = c("itemID_1", "itemID_2"))
rm(fuzzy)

fuzzy_clean <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set4b_fuzzy_clean.fthr", sep = "" ))
completeDate <- left_join(completeDate, fuzzy_clean, by = c("itemID_1", "itemID_2"))
rm(fuzzy_clean)

alternate <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set4c_alternate.fthr", sep = "" ))
completeDate <- left_join(completeDate, alternate, by = c("itemID_1", "itemID_2"))
rm(alternate)

similarity <- read_feather(paste(cache_loc, "/features_",trainOrTest, "_set4d_similarity.fthr", sep = "" ))
completeDate <- left_join(completeDate, similarity, by = c("itemID_1", "itemID_2"))
rm(similarity)
gc()

print("Saving Final Files")
write_feather(completeDate, paste("cache/final_featureSet_", trainOrTest, ".fthr", sep = "" ))
print("DONE")







