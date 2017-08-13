library(tidyverse)
library(feather)
library(xgboost)

nthread <- parallel::detectCores()

df_all <- read_feather("../input/df_all.fthr")
TARGET <- "noOfLanes"
NAString <- NA
model_features <- setdiff(names(df_all), c("roadId", TARGET, "data"))

df_all_train <- df_all[df_all$data == "train", ]
df_all_test <- df_all[df_all$data == "test", ]
#rm(df_all)
gc()

####### XGBOOST ############
EARLY_STOPPING <- 100
print.every.n <- 10
df_all_train[[TARGET]] <- as.factor(df_all_train[[TARGET]] - 1)
num_class <- length(levels(df_all_train[[TARGET]]))

param <- list(
  objective           = "multi:softprob", 
  booster             = "gbtree",
  eval_metric         = "mlogloss",
  num_class           = num_class,
  eta                 = 0.1,
  max_depth           = 5,
  subsample           = 0.9,
  min_child_weight    = 1,
  colsample_bytree    = 1.0,
  gamma               = 0,
  nthread             = nthread,
  num_parallel_tree   = 2
)

if (param$eval_metric != "auc") {
  isMaximize  <- F
} else {
  isMaximize  <- T
}
nrounds <- 100
seed <- (1:10)*1000

dtrain <- xgb.DMatrix(  data = data.matrix(df_all_train[, model_features]),
                        label = data.matrix(df_all_train[[TARGET]]),
                        missing = NAString)
watchlist <- list(train = dtrain)

t <- Sys.time()
print(param)
test_xgb_model <- rep(0, nrow(df_all_test))
for (s in seed) {
  cat("Generating XGB seed", s, "\n", sep = " ")
  set.seed(s)
  bst <- xgb.train(             params              = param,
                                data                = dtrain,
                                nrounds             = nrounds,
                                verbose             = 1,
                                print_every_n       = print.every.n,
                                early_stopping_rounds    = EARLY_STOPPING,
                                watchlist           = watchlist,
                                maximize            = isMaximize
  )
  print(format(Sys.time() - t, format = "%H:%M") )
  dtest <- xgb.DMatrix(  data = data.matrix(df_all_test[, model_features]),
                          missing = NAString)
  tmp <- predict(bst, dtest)
  tmp <- ifelse(tmp < 0, 0, tmp)
  test_xgb_model <- test_xgb_model + tmp
}
xgb_1 <- test_xgb_model / length(seed)


xgb_1 <- apply(matrix(xgb_1, byrow = T, ncol = num_class), 1, which.max)
xgb_1 <- data.frame(roadId = df_all_test$roadId, noOfLanes = xgb_1)
write_csv(xgb_1, "../output/finalSubmission.csv")

