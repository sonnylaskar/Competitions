#(c) Sonny Laskar (sonnylaskar at gmail Dot Com)
library(xgboost)
library(gbm)
library(randomForest)
library(caret)
library(readr)
library(dplyr)

doBuildCV <- F
dataURL <- 'http://az754797.vo.msecnd.net/competition/whra/data/WomenHealth_Training.csv'
seedForCV <- 1000

# Read data to R workspace. The string field religion is read as factors
colclasses <- rep("integer",50)
colclasses[36] <- "character"
dataset1 <- read.table(dataURL, header=TRUE, sep=",", strip.white=TRUE, stringsAsFactors = F, colClasses = colclasses)
#summary(dataset1)
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
  dat$patientID <- NULL
  dat$INTNR <- NULL
  dat <- manual_encode_religion(dat)
  dat$combine <- paste(dat$segment, dat$subgroup, sep = "")
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

#XGB Model
run_XGB <- function(dat, TARGET, GEO, DropList, seed = 1000, nrounds, BuildCV = T, Importance = F) {
  #Input: dat = data.frame
  #Input: TARGET Variable
  #Input: DropList = Columns to Drop
  #Input: nround = # of rounds
  #Input: BuildCV = Logical (Whether to perform CV)
  for (i in DropList) {
    dat[[i]] <- NULL
  }
  cat("GEO = ", GEO, "\n")
  dat <- dat[dat$geo == GEO, ]
  set.seed(seedForCV)
  #inTrain <- createDataPartition(y = paste(dat[[TARGET]]), p = 0.7, list = FALSE)
  inTrain <- createDataPartition(y = paste(dat[[TARGET]], dat$geo), p = 0.7, list = FALSE)
  
  dat[[TARGET]] <- as.factor(dat[[TARGET]])
  classes <- levels(dat[[TARGET]])
  levels(dat[[TARGET]]) <- c(0:(length(levels(dat[[TARGET]])) - 1))
  X_train <- dat[, names(dat) != TARGET]
  Y_train <- dat[[TARGET]]
  gc()
  features <- names(X_train)
  
  EVAL_METRIC <- "merror"
  OBJECTIVE <- "multi:softprob"
  BOOSTER <- "gbtree"
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
    num_class           = length(levels(Y_train)),
    nthread             = nthread,
    num_parallel_tree   = 1
  )
  
  
  if (BuildCV) {
    X_trainSet <- X_train[inTrain, ]
    Y_trainSet <- Y_train[inTrain]
    X_valSet <- X_train[-inTrain, ] #Replace ! with - if using index
    Y_valSet <- Y_train[-inTrain]
    
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
    #nrounds <- 100000
    
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
                                #early.stop.round    = EARLY_STOPPING,
                                #watchlist           = watchlist,
                                maximize            = isMaximize
  )
    print(bst$bestScore)
  print(format(Sys.time() - t, format = "%H:%M") )
  
  if (BuildCV) {
    print(bst$bestScore)
    print(bst$bestInd)
    #val_target <- getinfo(dval, 'label')
    val_target_xgb <- predict(bst, dval, missing=NA)
    probs <- as.data.frame(matrix(val_target_xgb, nrow=nrow(X_valSet), byrow = TRUE))
    colnames(probs) <- classes
    levels(Y_valSet) <- classes
    
    return(cbind(O = Y_valSet, probs))
    #return(cbind(G = X_valSet$geo, O = Y_valSet, probs))
  } else {
    modelfile <- paste("xgb_geo_", GEO, "_seed", seed, ".model", sep = "")
    print("Saving Model")
    saveRDS(bst, modelfile )
  }
  if (Importance) {
    print("Generating Importance File")
    imp <- xgb.importance(features, model = bst)
    importancefile <- paste("xgb_importance", TARGET, "csv", sep=".")
    write_csv(imp, importancefile)
  }
  
  
}

ETA <- 0.05
MAX_DEPTH <- 7
SUB_SAMPLE <- 0.7
MIN_CHILD_WEIGHT <- 1
COL_SAMPLE <- 0.8
GAMMA <- 0

seed <- 1000
sub <- data.frame()
for (GEO in 1:9) {
	xgb <- run_XGB(dataset1, 
                       TARGET = "combine", 
                       GEO = GEO,    
                       DropList = NULL, 
                       seed = seed,
                       nrounds = 320, 
                       BuildCV = doBuildCV, 
                       Importance = F)
	sub <- bind_rows(sub, xgb)
	if (doBuildCV) assign(paste("xgb", "geo", GEO, seed, sep = "_") , xgb)
}
final <- sub
final[is.na(final)] <- 0

final$NEW <- apply(final[, -1], 1, function(x) {
						m <- which.max(x)
						names(final)[-1][m]
						})
print("#################################################################")
print(sum(final$NEW == final$O) / nrow(final))
xgb <- final


####################
run_RF <- function(dat, TARGET, GEO, DropList, seed = 1000, nrounds, BuildCV = T, Importance = F) {
  #Input: dat = data.frame
  #Input: TARGET Variable
  #Input: DropList = Columns to Drop
  #Input: nround = # of rounds
  #Input: BuildCV = Logical (Whether to perform CV)
  for (i in DropList) {
    dat[[i]] <- NULL
  }
  cat("GEO = ", GEO, "\n")
  dat <- dat[dat$geo == GEO, ]
  set.seed(seedForCV)
  inTrain <- createDataPartition(y = paste(dat[[TARGET]], dat$geo), p = 0.7, list = FALSE)

  dat[[TARGET]] <- as.factor(dat[[TARGET]])
  classes <- levels(dat[[TARGET]])
  levels(dat[[TARGET]]) <- c(0:(length(levels(dat[[TARGET]])) - 1))
  features <- names(dat)
  
  
  if (BuildCV) {
    valSet <- dat[-inTrain, ] 
    train <- dat[inTrain, ]
  } else {
    train <- dat
  }

	for (s in seed) {
  	t <- Sys.time()
  	cat("Generating RF", "\n")
	set.seed(s)
  	model.rf = randomForest(combine ~ .,
                          data = train,
                          ntree=ntree,
                          mtry=mtry,
                          importance=F)
	assign(paste("model.rf", s, sep = ".") , model.rf)
  	print(format(Sys.time() - t, format = "%H:%M") )
  }
  
  if (BuildCV) {
    val_target_rf <- data.frame()
    for (s in seed) {
    	val_target_rf_temp = as.data.frame(predict(get(paste("model.rf", s, sep = ".")),
                            valSet,
                            type="prob"))
	if (nrow(val_target_rf) == 0 ) {
		val_target_rf <- val_target_rf_temp
	} else {
		val_target_rf  <- (val_target_rf + val_target_rf_temp) 
	}
    }
    val_target_rf <- val_target_rf / length(seed)
    colnames(val_target_rf) <- classes
    levels(valSet[[TARGET]]) <- classes
    return(cbind(O = valSet[[TARGET]], val_target_rf))
  } else {
    modelfile <- paste("rf_geo_", GEO, "_seed", seed, ".model", sep = "")
    print("Saving Model")
    saveRDS(model.rf, modelfile )
  }
  if (Importance) {
    print("Generating Importance File")
    imp <- xgb.importance(features, model = bst)
    importancefile <- paste("xgb_importance", TARGET, "csv", sep=".")
    write_csv(imp, importancefile)
  }
  
  
}
mtry <- 32 
ntree <- 400
seed <- c(1000)
sub <- data.frame()
for (GEO in 1:9) {
	rf <- run_RF(dataset1, 
	       GEO = GEO,
               TARGET = "combine", 
               DropList = NULL, 
               seed = seed,
               BuildCV = doBuildCV, 
               Importance = F)
	sub <- bind_rows(sub, rf)
}
final <- sub
final[is.na(final)] <- 0
final$NEW <- apply(final[, -1], 1, function(x) {
                                                m <- which.max(x)
                                                names(final)[-1][m]
                                                })

s <- (sum(final$NEW == final$O) / nrow(final))
print("#################################################################")
print(s)
rf <- final

####################
#Build a weighted score
w_xgb <- 0.4
w_rf <- 0.6
#w_gbm <- 1/3
final <- xgb[, -c(1,ncol(xgb))] * w_xgb + rf[, -c(1,ncol(xgb))] * w_rf #+ gbm[, -1] * w_gbm
final$NEW <- apply(final, 1, function(x) {
                                                m <- which.max(x)
                                                names(final)[m]
                                                })

final$ORIGINAL <- xgb$O
#head(final)
print("#################################################################")
print(sum(final$NEW == final$ORIGINAL) / nrow(final))

