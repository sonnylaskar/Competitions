library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(pROC)
library(caret)

#MODEL DESCRIPTION
#XGBOOST MODEL SEED = 500 and NROUND = 710

#LOAD DATA
train <- read.csv("../data/train_processes.csv", header = TRUE, stringsAsFactors = FALSE)
test <- read.csv("../data/test_processes.csv", header = TRUE, stringsAsFactors = FALSE)

#DONT NEED THESE COLUMNS ANY MORE
train$Earliest_Start_Date <- NULL
train$Internship_deadline <- NULL
train$Start_Date <- NULL
train$End_Date <- NULL
train$End.Date <- NULL
train$Start.Date <- NULL

test$Earliest_Start_Date <- NULL
test$Internship_deadline <- NULL
test$Start_Date <- NULL
test$End_Date <- NULL
test$End.Date <- NULL
test$Start.Date <- NULL

#Validation Set
set.seed(123)
inTrain <- createDataPartition(y = train$Is_Shortlisted, p = .70, list = FALSE)
trainSet <- train[inTrain, ]
validateSet <- train[-inTrain, ]
#####

dtrain <- xgb.DMatrix(data = data.matrix(train[, c(2:ncol(train))]), 
                      label = data.matrix(train$Is_Shortlisted),
                      missing=NA)
dvalidate <- xgb.DMatrix(data = data.matrix(validateSet[, c(2:ncol(validateSet))]), 
                         label = data.matrix(validateSet$Is_Shortlisted),
                         missing=NA)
watchlist <- list(train = dtrain, test = dvalidate)
param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "eta" = 0.1,
              "max_depth" = 10,
              "subsample" = 1,
              "min_child_weight" = 1,
              "colsample_bytree" = 0.2
              )
cv.nround <- 710

t <- Sys.time()
set.seed(500)
bst <- xgb.train(param = param,
                 data = dtrain,
                 nrounds = cv.nround,
                 maximize = TRUE)
print(Sys.time() - t)


test_target_xgb <- predict(bst, 
                               data.matrix(test[, c(2:ncol(test))]), 
                               missing=NA)
submission <- data.frame(Internship_ID = test$Internship_ID,
                         Student_ID = test$Student_ID,
                         Is_Shortlisted = test_target_xgb)
write_csv(submission,"../Submissions/XGB_MODEL_S500_N710.csv")  
