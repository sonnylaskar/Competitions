################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1e_attribute.R
# Description: This Rscript generates all Attribute (Json) features
# Usage:
#       Rscript ./code/3_feature_set1e_attribute.R train
#       Rscript ./code/3_feature_set1e_attribute.R test
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
library(jsonlite)

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



#Function to generate Attribute Features
attribute_feature <- function(w) {
	x <- w[1]
	y <- w[2]
	if (is.na(x) | is.na(y) | x == "[]" | y == "[]") {
		return(rep(NA,8))
	}
	x <- paste("[", x, "]", sep = "")
	y <- paste("[", y, "]", sep = "")
	x.df <- fromJSON(x, simplifyDataFrame = TRUE)
	y.df <- fromJSON(y, simplifyDataFrame = TRUE)
	N_Attr_x <- ncol(x.df)
	N_Attr_y <- ncol(y.df)
	if (N_Attr_x == 0 | N_Attr_y == 0) {
		return(rep(NA,8))
	}
	L <- length(intersect(names(x.df), names(y.df)))
	ratioOfPercentageOfMatchingAttributesNames <- L / min(N_Attr_x, N_Attr_y)
	ratioOfPercentageOfMatchingAttributesValues <- NA
	c <- 0
	if (ratioOfPercentageOfMatchingAttributesNames > 0) {
		for (i in intersect(names(x.df), names(y.df))) {
			if (x.df[[i]] == y.df[[i]]) {
				c <- c + 1
			}
		}
	ratioOfPercentageOfMatchingAttributesValues <- c / L
	}
	numberOfAttributes_sum <- N_Attr_x + N_Attr_y
	numberOfAttributes_diff <- abs(N_Attr_x - N_Attr_y)
	numberOfAttributes_min <- min(N_Attr_x, N_Attr_y)
	numberOfAttributes_max <- max(N_Attr_x, N_Attr_y)

	return(c(
			numberOfAttributes_sum, 
			numberOfAttributes_diff, 
			numberOfAttributes_min,
			numberOfAttributes_max,
			L, 
			ratioOfPercentageOfMatchingAttributesNames, 
			c, 
			ratioOfPercentageOfMatchingAttributesValues
		))
	

}

print("Generating Features")
#This can be made Parallel , I didnt do that as of now
df_master <- as.data.frame(t(apply(dat[, c("cleanjson_1", "cleanjson_2")], 1, attribute_feature)))
names(df_master) <- c(
			"numberOfAttributes_sum", 
			"numberOfAttributes_diff", 
			"numberOfAttributes_min",
			"numberOfAttributes_max",
			"NoOfMatchingAttributesNames", 
			"ratioOfPercentageOfMatchingAttributesNames", 
			"NoOfMatchingAttributesValues", 
			"ratioOfPercentageOfMatchingAttributesValues"
			)

names(df_master) <- paste("set1e", names(df_master), sep = "_")

######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Attributes features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1e_", "attributes.fthr", sep = "" ))

#END
