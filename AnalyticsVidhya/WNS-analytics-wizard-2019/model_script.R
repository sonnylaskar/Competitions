library(tidyverse)
library(lubridate)
library(zoo)
library(xgboost)
library(lightgbm)

train <- read_csv("../input/train.csv")
item_data <- read_csv("../input/item_data.csv")
view_log <- read_csv("../input/view_log.csv") %>%
  left_join(item_data, by = "item_id")
test <- read_csv("../input/test.csv")
windowPeriod <- 100
lagPeriod <- 7

createUserStats <- function() {
  view_log %>%
    mutate(date = date(server_time)) %>%
    group_by(date, user_id) %>%
    summarise(f_user_totalHits = n(),
              f_user_totalSessionCount = n_distinct(session_id),
              f_user_totalDeviceCount = n_distinct(device_type),
              f_user_totalItemCount = n_distinct(item_id),
              f_user_meanItemPrice = mean(item_price),
              f_user_totalItemPrice = sum(item_price),
              f_user_totalCategory1Count = n_distinct(category_1),
              f_user_totalCategory2Count = n_distinct(category_2),
              f_user_totalCategory3Count = n_distinct(category_3),
              f_user_totalProductTypeCount = n_distinct(product_type)
              ) %>%
    ungroup()
}

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))


createUserLagStats <- function() {
    userStats %>%
      group_by(user_id) %>%
      complete(date = full_seq(c(date, as.Date("2018-12-18")), period = 1)) %>%
      mutate(f_user_totalHits = rollapplyr(lag(f_user_totalHits, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalSessionCount = rollapplyr(lag(f_user_totalSessionCount, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalDeviceCount = rollapplyr(lag(f_user_totalDeviceCount, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalItemCount = rollapplyr(lag(f_user_totalItemCount, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_meanItemPrice = rollapplyr(lag(f_user_meanItemPrice, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalItemPrice = rollapplyr(lag(f_user_totalItemPrice, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalCategory1Count = rollapplyr(lag(f_user_totalCategory1Count, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalCategory2Count = rollapplyr(lag(f_user_totalCategory2Count, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalCategory3Count = rollapplyr(lag(f_user_totalCategory3Count, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T),
             f_user_totalProductTypeCount = rollapplyr(lag(f_user_totalProductTypeCount, lagPeriod), width = windowPeriod, FUN = mean, partial = TRUE, align = "right", fill = 0, na.rm = T))

}

userStats <- createUserStats()
userLagStats <- createUserLagStats()
userLagStats[is.nan(userLagStats)] <- 0
df_all <- bind_rows(train, test)
df_all$data <- ifelse(is.na(df_all$is_click), "test", "train")
df_all$date <- date(df_all$impression_time)
df_all <- left_join(df_all, userLagStats, by = c("date", "user_id"))
rm(userLagStats, userStats)
gc()
df_all$f_is_4G <- df_all$is_4G
df_all$f_app_code <- df_all$app_code
df_all <- df_all %>%
  group_by(user_id, date) %>%
  mutate(f_user_impression_rollingCount = seq(n())) %>%
  ungroup()
df_all <- df_all %>%
  group_by(user_id, date, app_code) %>%
  mutate(f_user_impressionPerApp_rollingCount = seq(n())) %>%
  ungroup()
df_all <- df_all %>%
  group_by(date, app_code) %>%
  mutate(f_user_impressionPerAppEachDate_rollingCount = seq(n())) %>%
  ungroup()


df_all$app_code <- as.character(df_all$app_code)
df_all$f_hour <- hour(df_all$impression_time)
df_all$f_wday <- wday(df_all$impression_time)
ohe_os <- as.data.frame(model.matrix(~.-1, select(df_all, os_version, app_code)))
names(ohe_os) <- paste("f", names(ohe_os), sep = "_")
df_all <- bind_cols(df_all, ohe_os)
rm(ohe_os)
gc()

nthread_all <- as.integer(future::availableCores())
nthread <- nthread_all

model_features <- grep("^f_", names(df_all), value = T)

train <- filter(df_all, data == "train")
test <- filter(df_all, data == "test")
test_X <- test[, model_features]
test_id <- test$impression_id
idx_train <- which(train$date <= as.Date("2018-12-06"))
train_X <- train[idx_train, model_features]
train_y <- train$is_click[idx_train]
val_X <- train[-idx_train, model_features]
val_y <- train$is_click[-idx_train]
gc()
nrounds <- 1000
EARLY_STOPPING <- 50
isMaximise <- T
seed <- c(1000)

param_lgb <- list(objective = "binary", # "regression",# "binary",
                  metric = "auc", # "auc",#   "binary_logloss", #   "binary_logloss",#  "auc",
                  max_depth = 5,
                  num_leaves = 10,
                  learning_rate = 0.1,
                  nthread = nthread,
                  bagging_fraction = 0.8,
                  feature_fraction = 0.7,
                  bagging_freq = 5,
                  bagging_seed = 1000,
                  verbosity = -1,
                  subsample= 0.8,
                  subsample_freq= 1,
                  colsample_bytree= 0.8,
                  min_child_weight= 0,
                  min_split_gain= 0,
                  min_data_in_leaf = 10)

param_xgb <- list(
  objective           = "binary:logistic", # "binary:logistic",
  booster             = "gbtree",
  eval_metric         = "auc", # "auc",
  eta                 = 0.1,
  max_depth           = 5,
  subsample           = 0.8,
  min_child_weight    = 1,
  colsample_bytree    = 0.8,
  gamma               = 0,
  nthread             = nthread,
  num_parallel_tree   = 1
)

lgb_val <- function(s, predict = T) {
  set.seed(s)
  dtrain <- lgb.Dataset(data = as.matrix(train_X),
                        label = train_y)
  dval <- lgb.Dataset(data = as.matrix(val_X),
                      label = val_y)
  valids <- list(val = dval)
  print("Building LGB Val Model")
  bst <- lgb.train(param_lgb,
                   data = dtrain,
                   nrounds = nrounds,
                   data_random_seed = s,
                   early_stopping_rounds = EARLY_STOPPING,
                   valids = valids)
  
  N_L <- bst$best_iter
  print(paste(c("LGB Val ", s, max(unlist(bst$record_evals$val$auc$eval)), bst$best_iter), sep = " "))
  
  if (!predict) return(NA)
  
  dtrain <- lgb.Dataset(data = as.matrix(bind_rows(train_X, val_X)),
                        label = c(train_y, val_y))
  bst <- lgb.train(param_lgb,
                   data = dtrain,
                   nrounds = N_L,
                   data_random_seed = s)
  
  #lgb.save(bst, MODELFILE)
  imp <- lgb.importance(bst)
  lgb.plot.importance(imp, top_n = 20)
  test_pred_lgb <- predict(bst, as.matrix(test_X), num_iteration = N_L)
  
  return(test_pred_lgb)

}

xgb_val <- function(s, predict = T) {
  dtrain <- xgb.DMatrix(  data = data.matrix(train_X),
                          label = data.matrix(train_y),
                          missing = NA)
  dval <- xgb.DMatrix(    data = data.matrix(val_X),
                          label = data.matrix(val_y),
                          missing = NA)
  watchlist <- list(val = dval)
  print("Building XGB Val Model")
  set.seed(s)
  bst <- xgb.train(             params              = param_xgb,
                                data                = dtrain,
                                nrounds             = nrounds,
                                verbose             = 1,
                                print_every_n       = 10,
                                early_stopping_rounds    = EARLY_STOPPING,
                                watchlist           = watchlist,
                                maximize            = isMaximise
  )
  
  
  N_X <- bst$best_iteration
  print(paste(c("XGB Val ", s, abs(bst$best_score), N_X), sep = " "))
  
  if (!predict) return(NA)
  
  dtrain <- xgb.DMatrix(  data = as.matrix(bind_rows(train_X, val_X)),
                          label = c(train_y, val_y),
                          missing = NA)
  set.seed(s)
  bst <- xgb.train(             params              = param_xgb,
                                data                = dtrain,
                                nrounds             = N_X,
                                verbose             = 1,
                                maximize            = isMaximise
  )
  
  test_pred_xgb <- predict(bst, as.matrix(test_X), num_iteration = N_X)
  
  return(test_pred_xgb)
  
}

test_pred_L_1000 <- lgb_val(1000, predict = F)
test_pred_X_1000 <- xgb_val(1000, predict = F)

test_pred_L_1000 <- lgb_val(1000, predict = T)
test_pred_X_1000 <- xgb_val(1000, predict = T)
test_pred_L_2000 <- lgb_val(2000, predict = T)
test_pred_X_2000 <- xgb_val(2000, predict = T)

sub <- tibble(impression_id = test_id,
              is_click = (test_pred_L_1000 + test_pred_L_2000 + test_pred_X_1000 + test_pred_X_2000) / 4)

write_csv(sub, "sub.csv")
