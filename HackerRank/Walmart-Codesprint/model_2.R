# Sonny Laskar (c)
library(readr)
library(text2vec)
library(dplyr)
library(caret)
library(xgboost)
library(gbm)
library(SnowballC)

train <- read_tsv("../input/train.tsv")
test <- read_tsv("../input/test.tsv")
train$'Item Class ID' <- as.numeric(train$'Item Class ID') #looks it should be numeric else bind_rows will not work

df_all <- bind_rows(train, test)
NROWS_TRAIN <- nrow(train)
rm(train, test)
gc()

#Function to clean HTML tags
cleanFun <- function(htmlString) {
  return(gsub("<.*?>", "", htmlString))
}

#Function to one hot encode
gen_feature_oneHot <- function(column, data) {
  
  deltaData <- select(data, -get(column))
  data <- select(data, get(column))
  dummies <- dummyVars(~ . -1, data = data)
  df2 <- predict(dummies, newdata = data)
  
  df2 <- cbind(deltaData, df2)
  return(df2)
}

tfIdf_Columns <- c("Actors",
                   "Actual Color",
                   "Product Long Description",
                   "Product Name",
                   "Product Short Description",
                   "Publisher",
                   "Recommended Use",
                   "Short Description",
                   "Synopsis")
for (i in tfIdf_Columns) {
  cat("TF IDf ", i, "\n")
  df_all[[i]]  <- cleanFun(df_all[[i]])

  stem_tokenizer <- function(x, tokenizer = word_tokenizer) {
    x %>%
      tokenizer %>%
      # poerter stemmer
      lapply(wordStem, 'en')
  }
  
  tokens <- df_all[[i]] %>%
    tolower %>%
    stem_tokenizer
  names(tokens) <- df_all$item_id
  
  stopwords <- c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours") %>%
    # here we stem stopwords, because stop-words filtering would be performed after tokenization!
    wordStem('en')
  it <- itoken(tokens)
  vocab <- create_vocabulary(it, ngram = c(1L, 2L), stopwords = stopwords)
  
  pruned_vocab = prune_vocabulary(vocab,  term_count_min = 50, doc_proportion_max = 0.4)
  
  it <- itoken(tokens)
  v_vectorizer <- vocab_vectorizer(pruned_vocab)
  dtm2 <- create_dtm(it, v_vectorizer)
  dtm2 <- as.data.frame(as.matrix(dtm2))
  print(dim(dtm2))

  if (exists("tf_idf") & ncol(dtm2) > 0) { 
    colnames(dtm2) <- paste(i, colnames(dtm2), sep = "_")
    tf_idf <- bind_cols(tf_idf, dtm2)
  } else if (ncol(dtm2) > 0) {
    tf_idf <- dtm2
  }
}

OneHotList <- c("Color",
                "Genre ID",
                "Literary Genre",
                "MPAA Rating",
                "Recommended Location",
                "Recommended Room",
                "actual_color")

for (i in OneHotList) {
  cat("One Hot Features ", i, "\n")
  df_all <- gen_feature_oneHot(i, df_all)
}

Numeric_ID_Fields <- c("Artist ID", "Item Class ID")
for (i in Numeric_ID_Fields) {
  cat("Numeric ", i, "\n")
  df_all[[i]] <- as.numeric(df_all[[i]])
}

LabelEncode_Fields <- c("Aspect Ratio", "Seller")
for (i in LabelEncode_Fields) {
  cat("Label Encode ", i, "\n")
  df_all[[i]] <- as.factor(df_all[[i]])
  levels(df_all[[i]]) <- c(1:length(levels(df_all[[i]])))
  df_all[[i]] <- as.numeric(df_all[[i]])
}

#Merge TFIDF 
df_all <- bind_cols(df_all, tf_idf)

#Split train and test
train <- df_all[1:NROWS_TRAIN, ]
test <- df_all[-(1:NROWS_TRAIN), ]
#rm(df_all)
#gc()

# What are the Shelves
SHELVES <- unique(gsub(" ", "", (unlist(strsplit(gsub("\\[|\\]", "", train$tag), ",")))))

TagToVector <- function(tag) {
  gsub(" ", "", unlist(strsplit(gsub("\\[|\\]", "", tag), ",")))
}

train$tag <- sapply(train$tag, TagToVector)

for (i in SHELVES) {
  cat("Add Tag Field ", i, "\n")  
  colName <- paste("Tag", i, sep = "_") 
  i <- paste("\\b", i, "\\b", sep = "") #Grepl needs those blocks for exact match
  train[[colName]] <- as.numeric(grepl(i, train$tag))
}

run_XGB <- function(train, TARGET, DropList, seed = 1000, nrounds, BuildCV = T, Importance = F) {
  #Input: dat = data.frame
  #Input: TARGET Variable
  #Input: DropList = Columns to Drop
  #Input: nround = # of rounds
  #Input: BuildCV = Logical (Whether to perform CV)
  
  X_train <- train #[, names(train) != TARGET]
  Y_train <- train[[TARGET]]
  
  
  set.seed(seedForCV)
  folds <- createFolds(Y_train, 3)  
  
  Y_train <- as.factor(Y_train)
  classes <- levels(Y_train)
  gc()
  
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
  
  
  if (BuildCV) {
    X_trainSet <- X_train[-folds[[fold]], ]
    Y_trainSet <- Y_train[-folds[[fold]]]
    X_valSet <- X_train[folds[[fold]], ] #Replace ! with - if using index
    Y_valSet <- Y_train[folds[[fold]]]
    
    rm(X_train)
    gc()
    print("Generating xgb.DMatrix")
    dtrain <- xgb.DMatrix(  data = data.matrix(X_trainSet),
                            label = data.matrix(Y_trainSet),
                            missing = NA)
    dval <- xgb.DMatrix(  data = data.matrix(X_valSet),
                          label = data.matrix(Y_valSet),
                          missing = NA)
    watchlist <- list(val = dval, train = dtrain)
    gc()
    nrounds <- 100000
  } else {
    print("Generating xgb.DMatrix")
    dtrain <- xgb.DMatrix(  data = data.matrix(X_train),
                            label = data.matrix(Y_train),
                            missing = NA)
    watchlist <- list(train = dtrain)
    rm(X_train, Y_train)
    gc()
  }
  
 
  t <- Sys.time()
  j <- 1
  cat("Generating XGB", "\n")
  set.seed(seed)
  
  bst <- xgb.train(             params              = param,
                                data                = dtrain,
                                nrounds             = nrounds,
                                verbose             = 1,
                                print.every.n       = print.every.n,
                                early.stop.round    = EARLY_STOPPING,
                                watchlist           = watchlist,
                                maximize            = isMaximize
  )
  print(bst$bestScore)
  print(format(Sys.time() - t, format = "%H:%M") )
  
  if (BuildCV) {
    print(bst$bestScore)
    print(bst$bestInd)
    val_target_xgb <- predict(bst, dval, missing=NA)
    probs <- as.data.frame(matrix(val_target_xgb, nrow=nrow(X_valSet), byrow = TRUE))
  } else {
    modelfile <- paste("xgb_seed", seed, TARGET, ".model", sep = "_")
    print(modelfile)
    print("Saving Model")
    saveRDS(bst, modelfile )
  }
  if (Importance) {
    print("Generating Importance File")
    imp <- xgb.importance(features, model = bst)
    importancefile <- paste("xgb_importance",  TARGET, "csv", sep=".")
    write_csv(imp, importancefile)
  }
  
}


fold <- 3
seedForCV <- 1000
DropList = c("item_id",
             "ISBN",
             "tag",
             "Actual Color",
             tfIdf_Columns,
             grep("Tag_", names(train), value = T))

ETA <- 0.1
MAX_DEPTH <- 1
SUB_SAMPLE <- 0.8
MIN_CHILD_WEIGHT <- 1
COL_SAMPLE <- 0.7
GAMMA <- 0
seed <- 1000
BOOSTER <- "gbtree" #  "gblinear" "gbtree"
doBuildCV <- F


#TARGET <- "Tag_4483"
for (TARGET in setdiff(grep("Tag_", names(train), value = T),
                       c("Tag_133270", "Tag_645319", "Tag_1229823"))) {
  print(TARGET)
  xgb <- run_XGB(train, 
                 TARGET = TARGET,
                 DropList = DropList,
                 seed = seed,
                 nrounds = 2000,
                 BuildCV = doBuildCV,
                 Importance = FALSE)
}
#library (ROCR)

predict_XGB <- function(test, DropList, TARGET, seed) {
  id <- test$item_id
  for (i in DropList) {
    cat("Dropping", i, "\n")
    test[[i]] <- NULL
  }
  
  modelfile <- paste("xgb_seed", seed, TARGET, ".model", sep = "_")
  print(modelfile)
  bst <- readRDS(modelfile )
  test_target_xgb <- predict(bst, data.matrix(test), missing=NA)

  final_test <- data.frame(item_id = id) 
  final_test[[TARGET]] <- test_target_xgb
  return(final_test)
}
rm(final_test)
for (TARGET in setdiff(grep("Tag_", names(train), value = T),
                       c("Tag_133270", "Tag_645319", "Tag_1229823"))) {
  print(TARGET)
  sub1000 <- predict_XGB(test, DropList, TARGET, 1000)
  
  if (exists("final_test")) {
    final_test <- left_join(final_test, sub1000, by = "item_id")
  } else {
    final_test <- sub1000
  }
}

#Create final Submission
finalSub <- data.frame(item_id = final_test$item_id)
finalSub$tag <- apply(final_test[, -1], 1, function(x) {
  idx <- which(x > 0.4)
  n <- names(final_test[, -1])[idx]
  if (length(n) == 0) {
    idx <- which(x > 0.3)
    n <- names(final_test[, -1])[idx]
  } 
  if (length(n) == 0) {
    idx <- which(x > 0.2)
    n <- names(final_test[, -1])[idx]
  } 
  n <- gsub("Tag_", "", n)
  paste("[", paste0( n, collapse = ", "), "]", sep = "")
})

write_tsv(finalSub, "Sub_21.tsv")

