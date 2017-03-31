library(readr)
library(dplyr)
library(caret)
library(xgboost)
library(gbm)
library(data.table)
library(lightgbm)
library(tm)
library(stringr)
library(ModelMetrics)

train <- fread("../input/train_indessa.csv", data.table = F)
test <- fread("../input/test_indessa.csv", data.table = F)

train$isTrain <- T
test$isTrain <- F

df_all <- bind_rows(train, test)

# Function to generate Binary Features
createBinaryFeatures <- function(dat, FilterList, field, N = 100, RANK = 0.6, STRL = 3) {
  x <- tolower(unlist(strsplit(dat[[field]], " ")))
  x <- x[!x %in% stopwords()]
  as.data.frame(table(x)) %>%
    arrange(-Freq) %>%
    mutate(rank = percent_rank(Freq)) %>%
    rowwise() %>%
    mutate(stringLength = nchar(as.character(x))) %>%
    filter(rank > RANK & stringLength >= STRL) %>%
    head(N) %>%
    select(x) %>%
    collect %>%
    .[["x"]] %>%
    tolower() %>%
    unique() %>%
    str_trim() -> FilterList
  for (value in FilterList) {
    cat(c("Adding binary fields on", field, value, "\n", sep = " "))
    colName <- paste(field, value, sep = "_")
    dat[[colName]] <- ifelse(grepl(value, dat[[field]], ignore.case = TRUE), 1, 0)
  }
  return(dat)
}

gen_feature_oneHot <- function(column, data) {
  
  deltaData <- select(data, -get(column))
  data <- select(data, get(column))
  dummies <- dummyVars(~ . -1, data = data)
  df2 <- predict(dummies, newdata = data)
  
  df2 <- cbind(deltaData, df2)
  return(df2)
}

featureEngg <- function(dat) {
  dat <- createBinaryFeatures(dat, FilterList, "title", N = 10)
  dat$f_length_emptitle <- nchar(dat$emp_title)
  dat$f_length_desc <- nchar(dat$desc)
  
  #Convert term to numeric
  dat$term <- as.numeric(gsub(" months", "", dat$term))
  
  dat$f_ratio_funded_amnt_inv_annualinc <- dat$funded_amnt_inv / dat$annual_inc
  dat$f_ratio_loadamnt_fundedamnt <- dat$funded_amnt / dat$loan_amnt
  dat$emp_length[dat$emp_length == "n/a"] <- -1
  dat$emp_length[dat$emp_length == "< 1 year"] <- 0
  dat$emp_length <- as.integer(unlist(str_extract_all(dat$emp_length,pattern = "\\d+")))

  dat$initial_list_status <- ifelse(dat$initial_list_status == "w", 1, 0)
  dat$application_type <- ifelse(dat$application_type == "JOINT", 1, 0)
  dat$last_week_pay <- as.numeric(gsub("th week", "", dat$last_week_pay))
  #OneHot Encode all Categorical variables
  OneHotList <- c("grade"
                  ,"sub_grade"
                  ,"home_ownership"
                  ,"verification_status"
                  ,"verification_status_joint"
                  ,"pymnt_plan"
                  ,"purpose"
                  #,"addr_state"
                  )
  for (i in OneHotList) {
    cat("One Hot Features ", i, "\n")
    dat <- gen_feature_oneHot(i, dat)
  }
  
  
  #Drop any columns with no variation
  for (i in names(dat)) {
    if (length(unique(dat[[i]])) <= 1) {
      cat("Dropping no variation column - ", i, "\n")
      dat[[i]] <- NULL
    }
  }

  #Drop these fields
  for (i in c("batch_enrolled",
              "zip_code",
              "emp_title",
              "desc",
              "addr_state",
              "title"
  )) {
    print(i)
    dat[[i]] <- NULL
  }

  #Replace space from names
  names(dat) <- gsub(" ", ".", names(dat))
  dat <- do.call(data.frame,lapply(dat, function(x) replace(x, is.infinite(x),NA)))
  #dat[is.na(dat)] <- -1  
  return(dat)
}

df_all <- featureEngg(df_all)

#Split back to train and test
train <- df_all[df_all$isTrain == T, ]
test <- df_all[df_all$isTrain == F, ]

TARGET = "loan_status"
DropList = c("isTrain",
             TARGET)

#seedForCV <- 1000
ETA <- 0.1
MAX_DEPTH <- 7
SUB_SAMPLE <- 0.9
MIN_CHILD_WEIGHT <- 1
COL_SAMPLE <- 0.9
GAMMA <- 0
seed <- 1000
BOOSTER <- "gbtree"
#nrounds <- 100

run_XGB <- function(train, TARGET, DropList, seed = 1000, nrounds) {
  X_train <- train
  
  Y_train <- train[[TARGET]]
  Y_train <- as.factor(Y_train)
  classes <- levels(Y_train)

  for (i in DropList) {
    cat("Dropping", i, "\n")
    X_train[[i]] <- NULL
  }
  features <- names(X_train)

  EVAL_METRIC <- "auc"
  OBJECTIVE <- "binary:logistic"
  BOOSTER <- BOOSTER
  nthread <- parallel::detectCores()
  isMaximize  <- T
  EARLY_STOPPING <- 100
  print_every_n <- 10
  param <- list(
    objective           = OBJECTIVE,
    booster             = BOOSTER,
    eval_metric         = EVAL_METRIC,
    eta                 = ETA,
    max_depth           = MAX_DEPTH,
    subsample           = SUB_SAMPLE,
    min_child_weight    = MIN_CHILD_WEIGHT,
    colsample_bytree    = COL_SAMPLE,
    gamma               = GAMMA,
    nthread             = nthread,
    num_parallel_tree   = 1
  )
  
  
  print("Generating xgb.DMatrix")
  dtrain <- xgb.DMatrix(  data = data.matrix(X_train),
                          label = data.matrix(Y_train),
                          missing = NA)
  watchlist <- list(train = dtrain)
  rm(X_train, Y_train)
  gc()
  
  t <- Sys.time()
  cat("Generating XGB", "\n")
  set.seed(seed)
  bst <- xgb.train(          params              = param,
                             data                = dtrain,
                             nrounds             = nrounds,
                             verbose             = 1,
                             print_every_n       = print_every_n,
                             early_stopping_rounds = EARLY_STOPPING,
                             watchlist           = watchlist,
                             maximize            = isMaximize
  )
  print(format(Sys.time() - t, format = "%H:%M") )

  for (i in DropList) {
    cat("Dropping", i, "\n")
    test[[i]] <- NULL
  }
  dtest <- xgb.DMatrix(data = data.matrix(test),
                       missing = NA)
  test_target_xgb <- predict(bst, dtest, missing=NA)
  probs <- as.data.frame(matrix(test_target_xgb, nrow=nrow(test), byrow = TRUE))
  names(probs) <- paste("xgb", seed, sep = "_")
  return(probs)
}
run_LightGBM <- function(train, TARGET, DropList, seed = 1000, nrounds) {
  X_train <- train
  
  Y_train <- train[[TARGET]]

  for (i in DropList) {
    cat("Dropping", i, "\n")
    X_train[[i]] <- NULL
  }
  features <- names(X_train)
  
  param <- list(num_leaves=4,
                max_depth = 7,
                min_data_in_leaf= 1,
                min_sum_hessian_in_leaf=100,
                learning_rate=0.1)
  print("Generating lgb.Dataset")
  dtrain <- lgb.Dataset(  data = data.matrix(X_train),
                          label = data.matrix(Y_train))
  
  #watchlist <- list(train = dtrain)
  rm(X_train, Y_train)
  gc()
  
  t <- Sys.time()
  cat("Generating Light GBM", "\n")
  #set.seed(seed)
  bst <- lgb.train(param, 
                   data = dtrain, 
                   nrounds = nrounds ,
                   #valids = watchlist,
                   obj = "binary",
                   data_random_seed = seed,
                   num_threads = 16)
  print(format(Sys.time() - t, format = "%H:%M") )
  
  for (i in DropList) {
    cat("Dropping", i, "\n")
    test[[i]] <- NULL
  }
  #dtest <- lgb.Dataset(  data = data.matrix(test))
  test_target_Lgbm <- predict(bst, data.matrix(test))
  probs <- as.data.frame(matrix(test_target_Lgbm, nrow=nrow(test), byrow = TRUE))
  names(probs) <- paste("lgbm", seed, sep = "_")
  return(probs)
}


xgb_1000 <- run_XGB(train,
               TARGET = TARGET,
               DropList = DropList,
               seed = 1000,
               nrounds = 1000)
xgb_2000 <- run_XGB(train,
                    TARGET = TARGET,
                    DropList = DropList,
                    seed = 2000,
                    nrounds = 1000)
xgb_3000 <- run_XGB(train,
                    TARGET = TARGET,
                    DropList = DropList,
                    seed = 3000,
                    nrounds = 1000)

xgb_final <- bind_cols(select(test, member_id), xgb_1000, xgb_2000, xgb_3000)
xgb_final$loan_status <- apply(xgb_final[, -1], 1, mean)
xgb_final$loan_status <- ifelse(xgb_final$loan_status == 1, 0.9999, xgb_final$loan_status)
xgb_final$loan_status <- ifelse(xgb_final$loan_status == 0, 0.0001, xgb_final$loan_status)


xgb_final <- xgb_final[, c("member_id", "loan_status")]
write_csv(xgb_final[, c("member_id", "loan_status")], "../output/final_sub.csv")

