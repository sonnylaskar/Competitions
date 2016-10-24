# This script is called by 3_feature_set1d_misc.R script
# DO NOT CALL it Directly
# Start Interaction feature script
print("Starting Interaction feature script")
featureList <- c("isMetroIdSame", "isLocationIDSame", "isRegionIDSame", "isLongitudeSame", "isLatitudeSame", "isTitleSame", "isdescriptionSame")
featureList <- combn(featureList, 2)

create_interaction <- function(x) {
        i <- x[1]
        j <- x[2]
        print(c(i, j))
        columnName <- paste("interaction", i, j, sep = "_")
        set1d[[columnName]] <<- ifelse(set1d[[i]] == 1 & set1d[[j]] == 1, 1, 0)
        return(NULL)
}
apply(featureList, 2, create_interaction)

set1d <- set1d[, grep("interaction", names(set1d), value = T)] #Filter only interaction features
names(set1d) <- paste("set1d", names(set1d), sep = "_")


######## Add Primary Columns ItemID1 and ItemID2
set1d <- cbind(dat[, grep("itemID_", names(dat), value = TRUE)], set1d)
print("Saving Interaction features features")
write_feather(set1d, paste(cache_loc, "/", "features_", trainOrTest, "_set1d_", "interaction.fthr", sep = "" ))

#END

