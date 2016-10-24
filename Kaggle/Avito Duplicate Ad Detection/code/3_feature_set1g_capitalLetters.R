################################################################################################
################################################################################################
#### Copyright (c) 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Competition: Avito Duplicate Ad Detection
# Filename : 3_feature_set1g_capitalLetters.R
# Description: This Rscript generates Capital Letters Features
# Usage:
#       Rscript ./code/3_feature_set1g_capitalLetters.R train
#       Rscript ./code/3_feature_set1g_capitalLetters.R test
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

# Read argument for train or test
trainOrTest <- commandArgs(trailingOnly = TRUE)
if (length(trainOrTest) > 1) {
        stop("ERROR: I need only 1 argument : train or test")
}

if (length(trainOrTest) == 0) {
        print("No Arguments passed, Assuming you mean test")
        trainOrTest <- "test"
}

#Load dat
FILENAME <- paste(cache_loc, "/", trainOrTest, ".csv", sep = "")
cat("Reading file ", FILENAME, "\n", sep = " ")
dat <- read_csv(FILENAME)


#Function to generate functions
getCapitalLetterFeatures <- function(x) {
	wordsWithCapitalLetters <- length(grep("[[:upper:]]", unlist(strsplit(x, " "))))
	countOfCapitalLetters <- length(grep("[[:upper:]]", unlist(strsplit(x, ""))))
	return(c(wordsWithCapitalLetters, countOfCapitalLetters))
}

df2 <- data.frame(ID = 1:nrow(dat)) #Else cbind will not work
for (Field in c("title_1", "title_2", "description_1", "description_2")) {
	print(Field)
	df2_temp <- as.data.frame(t(mcmapply(getCapitalLetterFeatures, dat[[Field]], USE.NAMES = F)))
	names(df2_temp) <- c(paste("wordsWithCapitalLetters", Field, sep = "_"), paste("countOfCapitalLetters", Field, sep = "_"))
	df2 <- cbind(df2, df2_temp)
}
for (i in c("title", "description")) {
	for (j in c("wordsWithCapitalLetters", "countOfCapitalLetters")) {
		#print(c(i,j))
		NewField1 <- paste(j, "_", i,"_1",  sep = "")
		NewField2 <- paste(j, "_", i,"_2",  sep = "")
		#print(c(NewField1,NewField2))
		NewFieldName <- paste("ratio", NewField1, NewField2, sep = "_")
		print(NewFieldName)
		df2[[NewFieldName]] <- df2[[NewField1]] / df2[[NewField2]]
		df2[[NewFieldName]] <- round(ifelse(df2[[NewFieldName]] > 1, 1/df2[[NewFieldName]], df2[[NewFieldName]]), 2)
	}
}

df2$ID <- NULL
names(df2) <- paste("set1g", names(df2), sep = "_")


######## Add Primary Columns ItemID1 and ItemID2
df2 <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], df2)
print("Saving Capital Letter  features")
write_feather(df2, paste(cache_loc, "/", "features_", trainOrTest, "_set1g_", "capitalLetters.fthr", sep = "" ))

#END
