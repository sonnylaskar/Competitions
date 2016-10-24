################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1f_SpecialCounting.R
# Description: This Rscript generates all Special Character Counting Features
# Usage:
#       Rscript ./code/3_feature_set1f_SpecialCounting.R train
#       Rscript ./code/3_feature_set1f_SpecialCounting.R test
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
library(stringr)
library(tm)

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



# Function to generate Features
getFeatures <- function(x, type) {
	if (type == "digit") {
		lengths((regmatches(x, gregexpr("[[:digit:]]+",x))))
	} else if (type == "cntrl") {
		lengths((regmatches(x, gregexpr("[[:cntrl:]]+",x))))
	} else if (type == "graph") {
                lengths((regmatches(x, gregexpr("[[:graph:]]+",x))))
        } else if (type == "punct") {
                lengths((regmatches(x, gregexpr("[[:punct:]]+",x))))
        } else if (type == "xdigit") {
                lengths((regmatches(x, gregexpr("[[:xdigit:]]+",x))))
        } else {
		return(NA)
	}
}

print("Generating Count Features")
for (i in c("digit", "graph", "punct", "xdigit")) {
	for (j in c("cleantitle_1", "cleantitle_2", "cleandesc_1", "cleandesc_2")) {
		print(c(i,j))
		assign(
			paste("countOf", i, "In", j , sep = "_"),
			sapply(dat[[j]], getFeatures, type = i, USE.NAMES = FALSE)
			)
	}
}

print("Generating Ratio Features")
for (i in c("_digit", "_graph_", "_punct_", "_xdigit_")) {
	for (j in c("title", "desc")) {
		print(c(i, j))
		f_name <- grep(i, grep(j, ls(), value = T), value = T)
		ratio <- get(f_name[1]) / get(f_name[2])
		ratio <- ifelse(ratio > 1, 1/ratio, ratio)
		assign(
                        paste("ratioOfcountOf", i, "In", j , sep = "_"),
			round(ratio, 2)
                        )
	}
}

df_master <- as.data.frame(do.call(cbind, list(sapply(grep("countOf", ls(), value = T), get, USE.NAMES = T))))
names(df_master) <- paste("set1f", names(df_master), sep = "_")

######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Special Counting features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1f_", "specialCounting.fthr", sep = "" ))

#END
