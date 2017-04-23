library(readr)
library(lubridate)
library(dplyr)
library(tidyr)
library(xgboost)

contact <- read_csv("../input/Train/Contacts_Pre_2017.csv")
resolution <- read_csv("../input/Train/Resolution_Pre_2017.csv")

for (i in c("Date", "End_Date")) {
  resolution[[i]] <- ymd(resolution[[i]])  
}

for (i in c("START.DATE", "END.DATE")) {
  contact[[i]] <- ymd(contact[[i]])  
}

contact$END.DATE <- NULL
contact <- contact %>%
  group_by(START.DATE, CONTACT.TYPE) %>%
  summarise(Contacts = sum(Contacts))

resolution <- select_(resolution, "Date", "Category", "Subject", "Resolution")
resolution <- resolution %>%
  group_by(Date, Category, Subject) %>%
  summarise(Resolution = sum(Resolution)) 

##################################################
#######   CONTACT
test_contact <- read_csv("../input/Test/Contacts2017.csv")
test_contact$Contacts <- NULL
df_all_contact <- expand.grid(seq(ymd('2010-01-01'),ymd('2016-12-31'),by='days'), 
            unique(contact$CONTACT.TYPE))

names(df_all_contact) <- names(test_contact)[1:2]
df_all_contact <- bind_rows(df_all_contact, test_contact)
df_all_contact <- left_join(df_all_contact, contact, by = c("Date" = "START.DATE",
                                                             "CONTACT.TYPE" = "CONTACT.TYPE"))
df_all_contact$Contacts <- ifelse(is.na(df_all_contact$Contacts), 0, df_all_contact$Contacts)

featureEngg_contact <- function(dat) {

  dat %>%
    arrange(Date) %>%
    group_by(CONTACT.TYPE) %>%
    mutate(f_Contacts_lag_75 = lag(Contacts, 75),
           f_Contacts_lag_90 = lag(Contacts, 90),
           f_Contacts_lag_120 = lag(Contacts, 120)) -> dat

  #Add Holiday
  holiday <- read_csv("../input/holiday.csv") 
  temp <- data.frame(Date = unique(dat$Date))
  temp <- left_join(temp, holiday, by = "Date")
  temp$f_Holiday <- ifelse(is.na(temp$f_Holiday), 0, temp$f_Holiday)
  temp <- temp %>%
    arrange(Date) %>%
    mutate(dummy = cumsum(f_Holiday)) %>%
    group_by(dummy) %>%
    mutate(f_days_since_last_holiday = 1:n() -1) %>%
    mutate(f_days_since_last_holiday=rank(f_days_since_last_holiday)/length(f_days_since_last_holiday)) %>% 
    ungroup() %>%
    select(-dummy)

  dat <- left_join(dat, temp, by = "Date")
  dat$f_Holiday <- ifelse(is.na(dat$f_Holiday), 0, dat$f_Holiday)

  dat$f_day <- day(dat$Date)
  dat$f_month <- month(dat$Date)
  #dat$f_week <- week(dat$Date)
  dat$f_wday <- wday(dat$Date)
  dat$f_quarter <- quarter(dat$Date)
  dat$CONTACT.TYPE <- as.factor(dat$CONTACT.TYPE)
  df2 <- as.data.frame(model.matrix(~CONTACT.TYPE-1, dat))
  names(df2) <- paste("f", names(df2), sep = "_")
  dat <- bind_cols(dat, df2)
  dat$CONTACT.TYPE <- NULL
  
  return(dat)
}

TARGET_Contact <- "Contacts"
df_all_contact <- featureEngg_contact(df_all_contact)
df_all_contact[is.na(df_all_contact)] <- -999
df_all_contact_train <- df_all_contact[df_all_contact$Date <= "2016-12-31", ]
df_all_contact_trainSet <- df_all_contact[df_all_contact$Date < "2016-09-01", ]
df_all_contact_valSet <- df_all_contact[df_all_contact$Date >= "2016-09-01" & df_all_contact$Date <= "2016-12-31", ]
df_all_contact_test <- df_all_contact[df_all_contact$Date > "2016-12-31", ]

model_features <- grep("f_", names(df_all_contact), value = T)

build_xgb_contact <- function() {
  ETA <- 0.06
  MAX_DEPTH <- 10
  SUB_SAMPLE <- 0.9
  MIN_CHILD_WEIGHT <- 1
  COL_SAMPLE <- 0.9
  GAMMA <- 0
  BOOSTER <- "gbtree"
  nrounds <- 2000
  seed <- (1:10)*1000
  
  EVAL_METRIC <- "rmse"
  OBJECTIVE <- "reg:linear"
  BOOSTER <- BOOSTER
  nthread <- parallel::detectCores()
  isMaximize  <- F
  EARLY_STOPPING <- 50
  print.every.n <- 10
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
  
  nrounds <- 50
  dtrain <- xgb.DMatrix(  data = data.matrix(df_all_contact_train[, model_features]),
                          label = data.matrix(df_all_contact_train[[TARGET_Contact]]),
                          missing = NA)
  watchlist <- list(train = dtrain)
  
  t <- Sys.time()
  print(param)
  test_xgb_contact <- rep(0, nrow(df_all_contact_test))
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
    dtest <- xgb.DMatrix(  data = data.matrix(df_all_contact_test[, model_features]),
                            missing = NA)
    tmp <- predict(bst, dtest, missing=NA)
    tmp <- ifelse(tmp < 0, 0, tmp)
    test_xgb_contact <- test_xgb_contact + tmp
  }
  test_xgb_contact <- test_xgb_contact / length(seed)
  sub_contact <- data.frame(ID = df_all_contact_test$ID, 
                            Contacts = test_xgb_contact)
  write_csv(sub_contact, "../output/Contacts.csv")
}
build_xgb_contact()

##################################################
#######   RESOLUTION

test_resolution <- read_csv("../input/Test/Resolution2017.csv")
test_resolution$Resolution <- NULL
df_all_resolution <- expand.grid(seq(ymd('2010-01-01'),ymd('2016-12-31'),by='days'), 
                                 unique(resolution$Category),
                                 unique(resolution$Subject))

names(df_all_resolution) <- names(test_resolution)[1:3]
df_all_resolution <- bind_rows(df_all_resolution, test_resolution)
df_all_resolution <- left_join(df_all_resolution, resolution, by = c("Date", "Category", "Subject"))
df_all_resolution$Resolution <- ifelse(is.na(df_all_resolution$Resolution), 0, df_all_resolution$Resolution)

featureEngg_resolution <- function(dat) {
  dat %>%
    arrange(Date) %>%
    group_by(Category, Subject) %>%
    mutate(f_Resolution_lag_75 = lag(Resolution, 75),
           f_Resolution_lag_90 = lag(Resolution, 90),
           f_Resolution_lag_120 = lag(Resolution, 120)) -> dat

  #Add Holiday
  holiday <- read_csv("../input/holiday.csv") 
  temp <- data.frame(Date = unique(dat$Date))
  temp <- left_join(temp, holiday, by = "Date")
  temp$f_Holiday <- ifelse(is.na(temp$f_Holiday), 0, temp$f_Holiday)
  temp <- temp %>%
    arrange(Date) %>%
    mutate(dummy = cumsum(f_Holiday)) %>%
    group_by(dummy) %>%
    mutate(f_days_since_last_holiday = 1:n() -1) %>%
    mutate(f_days_since_last_holiday=rank(f_days_since_last_holiday)/length(f_days_since_last_holiday)) %>% 
    ungroup() %>%
    select(-dummy)
  
  dat <- left_join(dat, temp, by = "Date")

  dat$f_day <- day(dat$Date)
  dat$f_month <- month(dat$Date)
  dat$f_week <- week(dat$Date)
  dat$f_wday <- wday(dat$Date)
  dat$f_quarter <- quarter(dat$Date)
  dat$Category <- as.factor(dat$Category)
  dat$Subject <- as.factor(dat$Subject)
  df2 <- as.data.frame(model.matrix(~.-1, dat[, c("Category", "Subject")]))
  names(df2) <- paste("f", names(df2), sep = "_")
  dat <- bind_cols(dat, df2)
  dat$Category <- NULL
  dat$Subject <- NULL
  
  return(dat)
}

df_all_resolution <- featureEngg_resolution(df_all_resolution)

TARGET <- "Resolution"
df_all_resolution[is.na(df_all_resolution)] <- -999
df_all_resolution_train <- df_all_resolution[df_all_resolution$Date <= "2016-12-31", ]
df_all_resolution_trainSet <- df_all_resolution[df_all_resolution$Date < "2016-09-01", ]
df_all_resolution_valSet <- df_all_resolution[df_all_resolution$Date >= "2016-09-01" & df_all_resolution$Date <= "2016-12-31", ]
df_all_resolution_test <- df_all_resolution[df_all_resolution$Date > "2016-12-31", ]

model_features <- grep("f_", names(df_all_resolution), value = T)

build_xgb_resolution <- function() {
  ETA <- 0.06
  MAX_DEPTH <- 6
  SUB_SAMPLE <- 0.9
  MIN_CHILD_WEIGHT <- 1
  COL_SAMPLE <- 0.9
  GAMMA <- 0
  BOOSTER <- "gbtree"
  nrounds <- 2000
  seed <- (1:10)*1000
  
  EVAL_METRIC <- "rmse"
  OBJECTIVE <- "reg:linear"
  BOOSTER <- BOOSTER
  nthread <- parallel::detectCores()
  isMaximize  <- F
  EARLY_STOPPING <- 50
  print.every.n <- 10
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
  nrounds <- 130
  dtrain <- xgb.DMatrix(  data = data.matrix(df_all_resolution_train[, model_features]),
                          label = data.matrix(df_all_resolution_train[[TARGET]]),
                          missing = NA)
  watchlist <- list(train = dtrain)
  
  t <- Sys.time()
  print(param)
  test_xgb_resolution <- rep(0, nrow(df_all_resolution_test))
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
    dtest <- xgb.DMatrix(  data = data.matrix(df_all_resolution_test[, model_features]),
                           missing = NA)
    tmp <- predict(bst, dtest, missing=NA)
    tmp <- ifelse(tmp < 0, 0, tmp)
    test_xgb_resolution <- test_xgb_resolution + tmp
  
  }
  test_xgb_resolution <- test_xgb_resolution / length(seed)
  
  sub_resolution <- data.frame(ID = df_all_resolution_test$ID, 
                               Resolution = test_xgb_resolution)
  write_csv(sub_resolution, "../output/Resolution.csv")
}
build_xgb_resolution()
