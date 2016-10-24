################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1i_images.R
# Description: This Rscript generates Image features
# Usage:
#       Rscript ./code/3_feature_set1i_images.R train
#       Rscript ./code/3_feature_set1i_images.R test
#       Default argument is test
################################################################################################
################################################################################################

args <- commandArgs(trailingOnly = F)
BASE <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))


# Source Config and functions.R file
source(paste(BASE, "/../config.cfg", sep = ""))
source(paste(BASE_DIR, "/code/functions.R", sep = ""))

preprocessing_nthreads <- 4

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
imageInfo <- arrange(imageInfo, image) 
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
names(dat)[ncol(dat)] <- c("imageSize1_1")
dat <- left_join(dat, imageInfo, by = c("images_array_1_2" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_2")
dat <- left_join(dat, imageInfo, by = c("images_array_1_3" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_3")
dat <- left_join(dat, imageInfo, by = c("images_array_1_4" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_4")
dat <- left_join(dat, imageInfo, by = c("images_array_1_5" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_5")
dat <- left_join(dat, imageInfo, by = c("images_array_1_6" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_6")
dat <- left_join(dat, imageInfo, by = c("images_array_1_7" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_7")
dat <- left_join(dat, imageInfo, by = c("images_array_1_8" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_8")
dat <- left_join(dat, imageInfo, by = c("images_array_1_9" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_9")
dat <- left_join(dat, imageInfo, by = c("images_array_1_10" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_10")
dat <- left_join(dat, imageInfo, by = c("images_array_1_11" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_11")
dat <- left_join(dat, imageInfo, by = c("images_array_1_12" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_12")
dat <- left_join(dat, imageInfo, by = c("images_array_1_13" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_13")
dat <- left_join(dat, imageInfo, by = c("images_array_1_14" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_14")
dat <- left_join(dat, imageInfo, by = c("images_array_1_15" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_15")
dat <- left_join(dat, imageInfo, by = c("images_array_1_16" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_16")
dat <- left_join(dat, imageInfo, by = c("images_array_1_17" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_17")
dat <- left_join(dat, imageInfo, by = c("images_array_1_18" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_18")
dat <- left_join(dat, imageInfo, by = c("images_array_1_19" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_19")
dat <- left_join(dat, imageInfo, by = c("images_array_1_20" = "image"))
names(dat)[ncol(dat)] <- c("imageSize1_20")


print("Joining COLUMNS for Set2")
dat <- left_join(dat, imageInfo, by = c("images_array_2_1" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_1")
dat <- left_join(dat, imageInfo, by = c("images_array_2_2" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_2")
dat <- left_join(dat, imageInfo, by = c("images_array_2_3" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_3")
dat <- left_join(dat, imageInfo, by = c("images_array_2_4" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_4")
dat <- left_join(dat, imageInfo, by = c("images_array_2_5" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_5")
dat <- left_join(dat, imageInfo, by = c("images_array_2_6" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_6")
dat <- left_join(dat, imageInfo, by = c("images_array_2_7" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_7")
dat <- left_join(dat, imageInfo, by = c("images_array_2_8" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_8")
dat <- left_join(dat, imageInfo, by = c("images_array_2_9" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_9")
dat <- left_join(dat, imageInfo, by = c("images_array_2_10" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_10")
dat <- left_join(dat, imageInfo, by = c("images_array_2_11" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_11")
dat <- left_join(dat, imageInfo, by = c("images_array_2_12" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_12")
dat <- left_join(dat, imageInfo, by = c("images_array_2_13" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_13")
dat <- left_join(dat, imageInfo, by = c("images_array_2_14" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_14")
dat <- left_join(dat, imageInfo, by = c("images_array_2_15" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_15")
dat <- left_join(dat, imageInfo, by = c("images_array_2_16" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_16")
dat <- left_join(dat, imageInfo, by = c("images_array_2_17" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_17")
dat <- left_join(dat, imageInfo, by = c("images_array_2_18" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_18")
dat <- left_join(dat, imageInfo, by = c("images_array_2_19" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_19")
dat <- left_join(dat, imageInfo, by = c("images_array_2_20" = "image"))
names(dat)[ncol(dat)] <- c("imageSize2_20")



dat <- unite(dat, "imageSize1", imageSize1_1, imageSize1_2, imageSize1_3, imageSize1_4, imageSize1_5, imageSize1_6, imageSize1_7, imageSize1_8, imageSize1_9, imageSize1_10, imageSize1_11, imageSize1_12, imageSize1_13, imageSize1_14, imageSize1_15, imageSize1_16, imageSize1_17, imageSize1_18, imageSize1_19, imageSize1_20
, sep = ",")
dat <- unite(dat, "imageSize2", imageSize2_1, imageSize2_2, imageSize2_3, imageSize2_4, imageSize2_5, imageSize2_6, imageSize2_7, imageSize2_8, imageSize2_9, imageSize2_10, imageSize2_11, imageSize2_12, imageSize2_13, imageSize2_14, imageSize2_15, imageSize2_16, imageSize2_17, imageSize2_18, imageSize2_19, imageSize2_20
, sep = ",")


generate_imageFeatures <- function(imageSize1, imageSize2) {
        images_array_1 <- unlist(strsplit(imageSize1, ","))
	images_array_1 <- as.integer(setdiff(images_array_1, "NA"))
        images_array_2 <- unlist(strsplit(imageSize2, ","))
	images_array_2 <- as.integer(setdiff(images_array_2, "NA"))

	if (length(images_array_1) == 0 | length(images_array_2) == 0) {
		return(c(rep(NA,3)))
	} 

	MinArray1 <- min(images_array_1)	
	MaxArray1 <- max(images_array_1)	
	MedianArray1 <- median(images_array_1)	

        MinArray2 <- min(images_array_2)
        MaxArray2 <- max(images_array_2)
        MedianArray2 <- median(images_array_2)

	ratioOfMinSize <- round(min(MinArray1, MinArray2) / max(MinArray1, MinArray2), 2)
	ratioOfMaxSize <- round(min(MaxArray1, MaxArray2) / max(MaxArray1, MaxArray2), 2)
	ratioOfMedianSize <- round(min(MedianArray1, MedianArray2) / max(MedianArray1, MedianArray2), 2)

        return( c(
		ratioOfMinSize, 
		ratioOfMaxSize,
		ratioOfMedianSize
		))
}


#Remove all columns which are not needed
dat <- dat[, c("itemID_1", "itemID_2", "imageSize1", "imageSize2") ]
rm(imageInfo)
gc()

print("Calculating Features")
t <- proc.time()
df_master <- as.data.frame(t(mcmapply(generate_imageFeatures, dat$imageSize1, dat$imageSize2, USE.NAMES = FALSE, mc.cores = preprocessing_nthreads)))
proc.time() - t
names(df_master) <- c("ratioOfMinSize", "ratioOfMaxSize", "ratioOfMedianSize")

dim(df_master)
dim(dat)

names(df_master) <- paste("set1i", names(df_master), sep = "_")
######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Image Hash features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1i_", "imageSize.fthr", sep = "" ))


