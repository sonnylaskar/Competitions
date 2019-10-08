library(tidyverse)
library(lubridate)
library(xgboost)
library(lightgbm)

train <- read_csv("../input/train.csv")
train$data <- "train"
item_data <- read_csv("../input/item_data.csv")
item_data <- as.data.frame(model.matrix(~.-1, item_data))

campaign_data <- read_csv("../input/campaign_data.csv")
campaign_data$start_date <- dmy(campaign_data$start_date)
campaign_data$end_date <- dmy(campaign_data$end_date)
campaign_data <- arrange(campaign_data, start_date)
campaign_data$campaign_type <- ifelse(campaign_data$campaign_type == "X", 1, 0)
  
coupon_item_mapping <- read_csv("../input/coupon_item_mapping.csv")

customer_demographics <- read_csv("../input/customer_demographics.csv")
customer_demographics$age_range <- as.numeric(factor(customer_demographics$age_range, 
                                          levels = c("18-25", "26-35", "36-45", "46-55", "56-70", "70+"),
                                          labels = c(18, 26, 36, 46, 56, 70)))
customer_demographics$marital_status <- ifelse(customer_demographics$marital_status == "Married", 1, 0)
customer_demographics$no_of_children <- as.numeric(ifelse(customer_demographics$no_of_children == "3+", 3, customer_demographics$no_of_children))
customer_demographics$family_size <- as.numeric(ifelse(customer_demographics$family_size == "5+", 5, customer_demographics$no_of_children))
#customer_transaction_data <- read_csv("../input/customer_transaction_data.csv")
test <- read_csv("../input/test_QyjYwdj.csv")
test$data <- "test"
#customer_transaction_df <- read_csv("../input/customer_transaction_df.csv")
customer_transaction_df <- read_csv("../input/agg_v2.csv")
#coupon_item_mapping$item_id[coupon_item_mapping$coupon_id == 27]
#customer_transaction_data[customer_transaction_data$item_id %in% coupon_item_mapping$item_id[coupon_item_mapping$coupon_id == 27] & 
#                            customer_transaction_data$customer_id == 1053,]

df_all <- bind_rows(train, test)
df_all <- left_join(df_all, coupon_item_mapping, by = c("coupon_id"))
df_all <- left_join(df_all, campaign_data, by = c("campaign_id"))
df_all <- left_join(df_all, customer_transaction_df, by = c("customer_id" = "customer_id",
                                                            "item_id" = "item_id",
                                                            "start_date" = "CampaignDate"))
#df_all <- left_join(df_all, customer_demographics, by = c("customer_id"))
#df_all <- left_join(df_all, item_data, by = c("item_id"))

#df_all$campaign_noofdays <- as.numeric(df_all$end_date - df_all$start_date)
rm(coupon_item_mapping, campaign_data, customer_transaction_df, customer_demographics, item_data)
gc()

nthread_all <- as.integer(future::availableCores())
nthread <- nthread_all

model_features <- setdiff(names(df_all), c("id", 
                                           "data",
                                           "campaign_id",
                                           "coupon_id",
                                           "customer_id",
                                           #"item_id",
                                           "start_date",
                                           "end_date",
                                           "redemption_status"
                                           ))

train <- filter(df_all, data == "train")
test <- filter(df_all, data == "test")
id <- test$id
test_X <- test[, model_features]
#test_id <- test$impression_id
idx_train <- which(train$campaign_id %in% 7:13)
train_X <- train[idx_train, model_features]
train_y <- train$redemption_status[idx_train]
val_X <- train[-idx_train, model_features]
val_y <- train$redemption_status[-idx_train]
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
  unique(test_pred_lgb)
  
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

sub <- tibble(id = id,
              redemption_status = test_pred_L_1000)
sub <- sub %>%
  group_by(id) %>%
  summarise(redemption_status = max(redemption_status))

write_csv(sub, "sub_5.csv")

