################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1h_images.R
# Description: This Rscript generates Image features
# Usage:
#       Rscript ./code/3_feature_set1h_images.R train
#       Rscript ./code/3_feature_set1h_images.R test
#       Default argument is test
################################################################################################
################################################################################################

args <- commandArgs(trailingOnly = F)
BASE <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))

# Source Config and functions.R file
source(paste(BASE, "/../config.cfg", sep = ""))
source(paste(BASE_DIR, "/code/functions.R", sep = ""))

preprocessing_nthreads <- 4 #Was facing some issues when I used more cores

#Load any additional packages
library(parallel)
library(stringdist)
library(tidyr)

options(scipen=999) #Since hammingDistnce is not working properly when represented in scientific notation

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

## Read Image Database
print("Reading Image file")
imageInfo <- read_csv(paste(cache_loc, "/", "image_database", ".csv", sep = ""))
imageInfo <- arrange(imageInfo, image) %>% 
		filter(FreqOfHash < 10) %>%
		select(image, ratioOfDimension, imagehash) #This is used for joining with master set
gc()
########################## TRAIN
print("Adding all new columns")
dat$images_array_1 <- gsub(" ", "", dat$images_array_1)
dat$images_array_2 <- gsub(" ", "", dat$images_array_2)

#generate upto 20 columns since 20 is the maximum number of images for any ad
dat <- separate(dat, images_array_1, paste("images_array_1", seq(1,20,1), sep = "_"), sep = ",")
dat <- separate(dat, images_array_2, paste("images_array_2", seq(1,20,1), sep = "_"), sep = ",")

for (x in grep("images_array", names(dat), value = T)) {
	dat[[x]] <- as.integer(dat[[x]])
}

print("Joining COLUMNS for Set1")
dat <- left_join(dat, imageInfo, by = c("images_array_1_1" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_1", "imagehash1_1")
dat <- left_join(dat, imageInfo, by = c("images_array_1_2" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_2", "imagehash1_2")
dat <- left_join(dat, imageInfo, by = c("images_array_1_3" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_3", "imagehash1_3")
dat <- left_join(dat, imageInfo, by = c("images_array_1_4" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_4", "imagehash1_4")
dat <- left_join(dat, imageInfo, by = c("images_array_1_5" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_5", "imagehash1_5")
dat <- left_join(dat, imageInfo, by = c("images_array_1_6" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_6", "imagehash1_6")
dat <- left_join(dat, imageInfo, by = c("images_array_1_7" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_7", "imagehash1_7")
dat <- left_join(dat, imageInfo, by = c("images_array_1_8" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_8", "imagehash1_8")
dat <- left_join(dat, imageInfo, by = c("images_array_1_9" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_9", "imagehash1_9")
dat <- left_join(dat, imageInfo, by = c("images_array_1_10" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_10", "imagehash1_10")
dat <- left_join(dat, imageInfo, by = c("images_array_1_11" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_11", "imagehash1_11")
dat <- left_join(dat, imageInfo, by = c("images_array_1_12" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_12", "imagehash1_12")
dat <- left_join(dat, imageInfo, by = c("images_array_1_13" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_13", "imagehash1_13")
dat <- left_join(dat, imageInfo, by = c("images_array_1_14" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_14", "imagehash1_14")
dat <- left_join(dat, imageInfo, by = c("images_array_1_15" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_15", "imagehash1_15")
dat <- left_join(dat, imageInfo, by = c("images_array_1_16" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_16", "imagehash1_16")
dat <- left_join(dat, imageInfo, by = c("images_array_1_17" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_17", "imagehash1_17")
dat <- left_join(dat, imageInfo, by = c("images_array_1_18" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_18", "imagehash1_18")
dat <- left_join(dat, imageInfo, by = c("images_array_1_19" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_19", "imagehash1_19")
dat <- left_join(dat, imageInfo, by = c("images_array_1_20" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension1_20", "imagehash1_20")

print("Joining COLUMNS for Set2")
dat <- left_join(dat, imageInfo, by = c("images_array_2_1" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_1", "imagehash2_1")
dat <- left_join(dat, imageInfo, by = c("images_array_2_2" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_2", "imagehash2_2")
dat <- left_join(dat, imageInfo, by = c("images_array_2_3" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_3", "imagehash2_3")
dat <- left_join(dat, imageInfo, by = c("images_array_2_4" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_4", "imagehash2_4")
dat <- left_join(dat, imageInfo, by = c("images_array_2_5" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_5", "imagehash2_5")
dat <- left_join(dat, imageInfo, by = c("images_array_2_6" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_6", "imagehash2_6")
dat <- left_join(dat, imageInfo, by = c("images_array_2_7" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_7", "imagehash2_7")
dat <- left_join(dat, imageInfo, by = c("images_array_2_8" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_8", "imagehash2_8")
dat <- left_join(dat, imageInfo, by = c("images_array_2_9" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_9", "imagehash2_9")
dat <- left_join(dat, imageInfo, by = c("images_array_2_10" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_10", "imagehash2_10")
dat <- left_join(dat, imageInfo, by = c("images_array_2_11" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_11", "imagehash2_11")
dat <- left_join(dat, imageInfo, by = c("images_array_2_12" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_12", "imagehash2_12")
dat <- left_join(dat, imageInfo, by = c("images_array_2_13" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_13", "imagehash2_13")
dat <- left_join(dat, imageInfo, by = c("images_array_2_14" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_14", "imagehash2_14")
dat <- left_join(dat, imageInfo, by = c("images_array_2_15" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_15", "imagehash2_15")
dat <- left_join(dat, imageInfo, by = c("images_array_2_16" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_16", "imagehash2_16")
dat <- left_join(dat, imageInfo, by = c("images_array_2_17" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_17", "imagehash2_17")
dat <- left_join(dat, imageInfo, by = c("images_array_2_18" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_18", "imagehash2_18")
dat <- left_join(dat, imageInfo, by = c("images_array_2_19" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_19", "imagehash2_19")
dat <- left_join(dat, imageInfo, by = c("images_array_2_20" = "image"))
names(dat)[(ncol(dat)-1):ncol(dat)] <- c("ratioOfDimension2_20", "imagehash2_20")


dat <- unite(dat, "imagehash1", imagehash1_1, imagehash1_2, imagehash1_3, imagehash1_4, imagehash1_5, imagehash1_6, imagehash1_7, imagehash1_8, imagehash1_9, imagehash1_10, imagehash1_11, imagehash1_12, imagehash1_13, imagehash1_14, imagehash1_15, imagehash1_16, imagehash1_17, imagehash1_18, imagehash1_19, imagehash1_20, sep = ",")
dat <- unite(dat, "imagehash2", imagehash2_1, imagehash2_2, imagehash2_3, imagehash2_4, imagehash2_5, imagehash2_6, imagehash2_7, imagehash2_8, imagehash2_9, imagehash2_10, imagehash2_11, imagehash2_12, imagehash2_13, imagehash2_14, imagehash2_15, imagehash2_16, imagehash2_17, imagehash2_18, imagehash2_19, imagehash2_20, sep = ",")


generate_imageFeatures <- function(imagehash1, imagehash2) {
        images_array_1 <- unlist(strsplit(imagehash1, ","))
	images_array_1 <- setdiff(images_array_1, "NA")
        images_array_2 <- unlist(strsplit(imagehash2, ","))
	images_array_2 <- setdiff(images_array_2, "NA")
	if (length(images_array_1) == 0) {
		isImageNullItem1 <- 1
	} else {
		isImageNullItem1 <- 0
	}
	if (length(images_array_2) == 0) {
		isImageNullItem2 <- 1
	} else {
		isImageNullItem2 <- 0
	}

	if (length(images_array_1) == 0 & length(images_array_2) == 0) {
		isImageNullBothItems <- 1
	} else {
		isImageNullBothItems <- 0
	}

	if (isImageNullItem1 == 1 | isImageNullItem2 == 1) {
		return(c(isImageNullItem1, isImageNullItem2, isImageNullBothItems, rep(NA,6)))
	}
	
        df2 <- expand.grid(images_array_1, images_array_2, stringsAsFactors = FALSE)

        df2 <- df2 %>% rowwise() %>% mutate(hammingDistance = round(stringdist(Var1, Var2, method = "hamming"), 2), cosineDistance = round(stringdist(Var1, Var2, method = "cosine"), 2))

	return(	c(
		isImageNullItem1,
		isImageNullItem2,
		isImageNullBothItems,
		round(min(df2$cosineDistance, na.rm = T), 2), 
		round(max(df2$cosineDistance, na.rm = T), 2), 
		round(median(df2$cosineDistance, na.rm = T), 2),  
		round(min(df2$hammingDistance, na.rm = T), 2), 
		round(median(setdiff(df2$hammingDistance, Inf), na.rm = T), 2), 
		round(sum(df2$hammingDistance == 0), 2)
		))
}

#Feature: Median ratioOfDimension
dat$medianRatioOfDimension1 <- round(apply(dat[, grep("ratioOfDimension1", names(dat), value = T)], 1, median, na.rm = T), 2)
dat$maxRatioOfDimension1 <- apply(dat[, grep("ratioOfDimension1", names(dat), value = T)], 1, max, na.rm = T)
dat$maxRatioOfDimension1 <- round(ifelse(is.infinite(dat$maxRatioOfDimension1), NA, dat$maxRatioOfDimension1), 2) #Since max will retun Inf also
dat$minRatioOfDimension1 <- apply(dat[, grep("ratioOfDimension1", names(dat), value = T)], 1, min, na.rm = T)
dat$minRatioOfDimension1 <- round(ifelse(is.infinite(dat$minRatioOfDimension1), NA, dat$minRatioOfDimension1), 2) #Since min will retun Inf also

dat$medianRatioOfDimension2 <- round(apply(dat[, grep("ratioOfDimension2", names(dat), value = T)], 1, median, na.rm = T), 2)
dat$maxRatioOfDimension2 <- apply(dat[, grep("ratioOfDimension2", names(dat), value = T)], 1, max, na.rm = T)
dat$maxRatioOfDimension2 <- round(ifelse(is.infinite(dat$maxRatioOfDimension2), NA, dat$maxRatioOfDimension2) , 2)#Since max will retun Inf also
dat$minRatioOfDimension2 <- apply(dat[, grep("ratioOfDimension2", names(dat), value = T)], 1, min, na.rm = T)
dat$minRatioOfDimension2 <- round(ifelse(is.infinite(dat$minRatioOfDimension2), NA, dat$minRatioOfDimension2), 2) #Since min will retun Inf also

dat$relativeMedianRatioOfDimension <- apply(dat[, c("medianRatioOfDimension1", "medianRatioOfDimension2")], 1, median, na.rm = T)

#Remove all columns which are not needed
dat <- dat[, c("itemID_1", "itemID_2", "imagehash1", "imagehash2", grep("RatioOfDimension", names(dat), value = T))]
rm(imageInfo)
gc()


print("Calculating Features")
t <- proc.time()
df_master <- as.data.frame(t(mcmapply(generate_imageFeatures, dat$imagehash1, dat$imagehash2, USE.NAMES = FALSE, mc.cores = preprocessing_nthreads)))
proc.time() - t
names(df_master) <- c("isImageNullItem1", "isImageNullItem2", "isImageNullBothItems", "image_minCosineDistance", "image_maxCosineDistance", "image_medianCosineDistance", "image_minHammingDistance", "image_medianHammingDistance", "image_countOfZeroHammingDistance")

dim(df_master)
dim(dat)
df_master <- bind_cols(df_master, dat[, grep("RatioOfDimension", names(dat), value = T)])

names(df_master) <- paste("set1h", names(df_master), sep = "_")
######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Image Hash features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1h_", "image.fthr", sep = "" ))

#END


