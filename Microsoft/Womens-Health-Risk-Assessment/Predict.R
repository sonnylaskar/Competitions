#install.packages("src/downloaded_packages/chron_2.3-47.zip", lib = ".", repos = NULL, verbose = TRUE)
install.packages("src/downloaded_packages/stringi_1.1.1.zip", lib = ".", repos = NULL, verbose = TRUE)
#install.packages("src/downloaded_packages/data.table_1.9.6.zip", lib = ".", repos = NULL, verbose = TRUE)
install.packages("src/downloaded_packages/magrittr_1.5.zip", lib = ".", repos = NULL, verbose = TRUE)
#install.packages("src/downloaded_packages/stringr_1.0.0.zip", lib = ".", repos = NULL, verbose = TRUE)
install.packages("src/downloaded_packages/xgboost_0.4-4.zip", lib = ".", repos = NULL, verbose = TRUE)

library(xgboost, lib.loc=".", verbose=TRUE)
library(dplyr)
library(gbm)
library(randomForest)
# Map 1-based optional input ports to variables
#dataset1 <- maml.mapInputPort(1) # class: data.frame
dataset1 <- read.csv("WomenHealth_Test.csv", header = T)
dataset1 <- read.csv("WomenHealth_Training.csv", header = T)
#dataset1 <- dataset1[1, ]
dataset1$segment <- NULL
dataset1$subgroup <- NULL
cat("Original dim: ", dim(dataset1), "\n")


encode_religion <- function(dat) {
  #Input: Character Vector for religion
  #Output: Numeric Vector
  dat <- ifelse(dat == "Buddhist", 1, dat)
  dat <- ifelse(dat == "Evangelical/Bo", 2, dat)
  dat <- ifelse(dat == "Hindu", 3, dat)
  dat <- ifelse(dat == "Jewish", 4, dat)
  dat <- ifelse(dat == "Muslim", 5, dat)
  dat <- ifelse(dat == "Other", 6, dat)
  dat <- ifelse(dat == "Other Christia", 7, dat)
  dat <- ifelse(dat == "Roman Catholic", 8, dat)
  dat <- ifelse(dat == "Russian/Easter", 9, dat)
  dat <- ifelse(dat == "Traditional/An", 10, dat)
  dat <- ifelse(dat == "", NA, dat)
  dat <- as.integer(dat)
  return(dat)
}

manual_encode_religion <- function(dat) {
  #Input: Character Vector for religion
  #Output: Numeric Vector
  RELIGION <- c("Hindu", "Evangelical/Bo", "Muslim", "Roman Catholic", "Other Christia", "Buddhist", "Russian/Easter", "Traditional/An", "Other", "Jewish")
  for (i in RELIGION) {
    c <- paste("religion", i, sep = ".")
    print(c)
    dat[[c]] <- ifelse(dat$religion == i, 1, 0)
  }
  dat$religion <- encode_religion(dat$religion)
  return(dat)
}

featureEngineering <- function(dat) {
  dat$INTNR <- NULL
  dat$geo <- as.integer(dat$geo)
  dat <- manual_encode_religion(dat)
  #dat$combine <- paste(dat$segment, dat$subgroup, sep = "")
  dat$segment <- NULL
  dat$subgroup <- NULL
  dat[is.na(dat)] <- -1
  dat$christian <- as.numeric(dat$christian) #Xgboost needs at least one column as numeric
  #Random Forest cannot handle / and space in colnames
  names(dat) <- gsub("/", "_", names(dat))
  names(dat) <- gsub(" ", "_", names(dat))

  return(dat)
}
dataset1 <- featureEngineering(dataset1)
cat("New dim: ", dim(dataset1), "\n")


sub <- data.frame(patientID = NULL, geo = NULL, class = NULL)
for (GEO in 1:9) {
	print(GEO)
	dat <- dataset1[dataset1$geo == GEO, ]
	cat("New dim: ", dim(dat), "\n")
	if (nrow(dat) == 0) next
	patientID <- dat$patientID
	dat$patientID <- NULL
	
	if (GEO == 1) classes <- c("11","21","22") 
	if (GEO == 2) classes <- c("11","12","21","22","31","41")
	if (GEO == 3) classes <- c("11","12","21","22")
	if (GEO == 4) classes <- c("11","12")
	if (GEO == 5) classes <- c("11","12","22","31","32")
	if (GEO == 6) classes <- c("11","12","21")
	if (GEO == 7) classes <- c("11","12","21","22","31")
	if (GEO == 8) classes <- c("11","21","31","41")
	if (GEO == 9) classes <- c("11","12","21","31","32")
	#LOAD XGB Model
	xgb_1000 <- readRDS(paste("src/downloaded_packages/xgb_geo_", GEO ,"_seed1000.model", sep = ""))

	xgb_test <- predict(xgb_1000, data.matrix(dat), missing=NA)
	xgb_test <- as.data.frame(matrix(xgb_test,
                              		nrow=nrow(dat),
	                              	byrow = TRUE))
	colnames(xgb_test) <- classes

	#LOAD RF Model
	rf_1000 <- readRDS(paste("src/downloaded_packages/rf_geo_", GEO ,"_seed1000.model", sep = ""))
	rf_test <- as.data.frame(predict(rf_1000,
                       		dat,
	                      	type= "prob"))
	colnames(rf_test) <- classes

	#Combined Weightage
	final <- (xgb_test*0.4 + rf_test*0.6)
	final$NEW <- apply(final, 1, function(x) {
                                                m <- which.max(x)
                                                names(final)[m]
                                                })
	sub <- rbind(sub, data.frame(patientID = patientID, geo = dat$geo, class = final$NEW))
}

data.set <- data.frame(patientID = sub$patientID,
                       Geo_Pred = sub$geo,
                       Segment_Pred = as.integer(substring(sub$class, 1, 1)),
                       Subgroup_Pred = as.integer(substring(sub$class, 2, 2))
			)

print(str(data.set))
maml.mapOutputPort("data.set");

