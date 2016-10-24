################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1a_ngram.R
# Description: This Rscript generates all ngram features
# Usage: 
#	Rscript ./code/3_feature_set1a_ngram.R train
#	Rscript ./code/3_feature_set1a_ngram.R test
#	Default argument is test
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

#######################################
# Start generating Features for DESCRIPTION columns
print("Start generating nGrams Features for DESCRIPTION columns")
for (n in 1:3) {
  print(n)
  df2 <- data.frame(t(mcmapply(getNgramsCount, dat$cleandesc_1, dat$cleandesc_2, n, USE.NAMES = FALSE, mc.cores = preprocessing_nthreads)))
  colnames(df2) <- c(
                        paste("countOf_", n, "_Grams_description_min", sep = ""),
                        paste("countOf_", n, "_Grams_description_max", sep = ""),
                        paste("countOf_", n, "_Grams_description_sum", sep = ""),
                        paste("countOf_", n, "_Grams_description_diff", sep = ""),

                        paste("countOf_", n, "_Grams_cleandesc_1", sep = ""),
                        paste("countOf_", n, "_Grams_cleandesc_2", sep = ""),
                        paste("countOfUnique_", n, "_Grams_cleandesc_1", sep = ""),
                        paste("countOfUnique_", n, "_Grams_cleandesc_2", sep = ""),
                        paste("ratioOf_", n, "_Grams_cleandesc_1_cleandesc_2", sep = ""),
                        paste("ratioOfUnique_", n, "_Grams_cleandesc_1_cleandesc_2", sep = ""),
                        paste("ratioOfIntersect_", n, "_Grams_cleandesc_1_in_cleandesc_2", sep = ""),
                        paste("ratioOfIntersect_", n, "_Grams_cleandesc_2_in_cleandesc_1", sep = "")
                    )
  if (nrow(df2) != nrow(dat)) {
	cat("Expecting", nrow(dat), "Got", nrow(df2), "\n", sep = " ")
	stop("mcmapply is behaving weird. Getting less results")
  }

  if (exists("df_master")) {
        df_master <- bind_cols(df_master, df2)
  } else {
        df_master <- df2
  }
}
names(df_master) <- paste("set1a", names(df_master), sep = "_")

######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Description ngrams features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1a_", "ngram_description.fthr", sep = "" ))
rm(df_master, df2)
gc()


#######################################
# Start generating Features for TITLE columns
print("Start generating nGrams Features for TITLE columns")
for (n in 1:3) {
  print(n)
  df2 <- data.frame(t(mcmapply(getNgramsCount, dat$cleantitle_1, dat$cleantitle_2, n, USE.NAMES = FALSE, mc.cores = preprocessing_nthreads)))
  colnames(df2) <- c(
                        paste("countOf_", n, "_Grams_title_min", sep = ""),
                        paste("countOf_", n, "_Grams_title_max", sep = ""),
                        paste("countOf_", n, "_Grams_title_sum", sep = ""),
                        paste("countOf_", n, "_Grams_title_diff", sep = ""),

                        paste("countOf_", n, "_Grams_cleantitle_1", sep = ""),
                        paste("countOf_", n, "_Grams_cleantitle_2", sep = ""),
                        paste("countOfUnique_", n, "_Grams_cleantitle_1", sep = ""),
                        paste("countOfUnique_", n, "_Grams_cleantitle_2", sep = ""),
                        paste("ratioOf_", n, "_Grams_cleantitle_1_cleantitle_2", sep = ""),
                        paste("ratioOfUnique_", n, "_Grams_cleantitle_1_cleantitle_2", sep = ""),
                        paste("ratioOfIntersect_", n, "_Grams_cleantitle_1_in_cleantitle_2", sep = ""),
                        paste("ratioOfIntersect_", n, "_Grams_cleantitle_2_in_cleantitle_1", sep = "")
                    )

  if (nrow(df2) != nrow(dat)) {
	cat("Expecting", nrow(dat), "Got", nrow(df2), "\n", sep = " ")
	stop("mcmapply is behaving weird. Getting less results")
  }

  if (exists("df_master")) {
        df_master <- bind_cols(df_master, df2)
  } else {
        df_master <- df2
  }
}
names(df_master) <- paste("set1a", names(df_master), sep = "_")

######## Add Primary Columns ItemID1 and ItemID2
df_master <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df_master)
print("Saving Title ngrams features")
write_feather(df_master, paste(cache_loc, "/", "features_", trainOrTest, "_set1a_", "ngram_title.fthr", sep = "" ))
rm(df_master, df2)
gc()

#END
