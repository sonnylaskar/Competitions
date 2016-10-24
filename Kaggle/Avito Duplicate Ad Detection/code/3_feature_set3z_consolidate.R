library(readr)
library(Hmisc)
library(raster)
library(xgboost)
library(textreuse)
library(jsonlite)
library(data.table)
library(igraph)
library(dummies)
library(feather)

args <- commandArgs(trailingOnly = F)
BASE <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))

# Source Config and functions.R file
source(paste(BASE, "/../config.cfg", sep = ""))

# Read argument for train or test
trainOrTest <- commandArgs(trailingOnly = TRUE)
if (length(trainOrTest) > 1) {
        stop("ERROR: I need only 1 argument : train or test")
}

if (length(trainOrTest) == 0) {
        print("No Arguments passed, Assuming you mean test")
        trainOrTest <- "test"
}
#################################


# Json Frequency erzeugen
JSON_FREQ <- paste(cache_loc, "/jfreq.RData", sep = "")
if (file.exists(JSON_FREQ)) {
	print("Loading JsonFreq File")
	load(JSON_FREQ)
} else {
  print("Generating JsonFreq File")
  # create JSON FREQ feature
  infotest <- read_csv(test_ItemInfo)
  infotest <- data.table(infotest[,c("itemID","attrsJSON","title","description")])
  
  # read data
  info <-  read_csv(train_ItemInfo)
  infotest <- rbind(infotest,data.table(info[,c("itemID","attrsJSON","title","description")]))
  setkey(infotest,attrsJSON)
  jfreq <-infotest[,.(.N),by=attrsJSON]
  colnames(jfreq) <- c("attrsJSON","jsonFreq")
  setkey(infotest,title)
  tfreq <-infotest[,.(.N),by=title]
  colnames(tfreq) <- c("title","titleFreq")
  setkey(infotest,description)
  dfreq <-infotest[,.(.N),by=description]
  colnames(dfreq) <- c("description","descFreq")
  
  rm(infotest)
  rm(info)
  save.image(file=JSON_FREQ)
}

# Cluster features 
CLUSTER_TRAIN <- paste(cache_loc, "/cluster_trainV2.csv", sep = "")
if (!file.exists(CLUSTER_TRAIN)) {
  print("Generating Cluster for Train")
  library(igraph)
  library(readr)
  library(Hmisc)
  # Train 
  #######
  
  train  <- read_csv(train_ItemPairs)
  
  # Generate clusterids
  gall<-graph_(cbind(train$itemID_1,train$itemID_2),from_edgelist(directed=FALSE)) 
  call<- clusters(gall,mode="strong")
  train$AllClusterID <- call$membership[train$itemID_1]
  
  # clusteroccurs
  aux <- aggregate(train$AllClusterID,by=list(train$AllClusterID),length)
  names(aux) <- c("AllClusterID","clusteroccurs")
  train <- merge(train,aux,by="AllClusterID",all.x=T)
  
  # clustersize 
  aux1 <- train[,c("AllClusterID","itemID_1")]
  aux2 <- train[,c("AllClusterID","itemID_2")]
  names(aux1) <- c("AllClusterID","itemID")
  names(aux2) <- c("AllClusterID","itemID")
  aux <- rbind(aux1,aux2)
  aux <- unique(aux)
  aux <- aggregate(aux$itemID,by=list(aux$AllClusterID),length)
  names(aux) <- c("AllClusterID","clustersize")
  train <- merge(train,aux,by="AllClusterID",all.x=T)
  
  # clustercover
  train$clustercover <- train$clusteroccurs*2 / (train$clustersize * (train$clustersize-1))
  write.csv(train[,c("itemID_1","itemID_2","clustersize","clusteroccurs","clustercover")],
            file=CLUSTER_TRAIN,row.names = F)
}

CLUSTER_TEST <- paste(cache_loc, "/cluster_testV2.csv", sep = "")
if (!file.exists(CLUSTER_TEST)) {
  
  print("Generating Cluster for Test")
  # Test
  #######
  
  train  <- read_csv(test_ItemPairs)
  
  # Generate clusterids
  gall<-graph_(cbind(train$itemID_1,train$itemID_2),from_edgelist(directed=FALSE)) 
  call<- clusters(gall,mode="strong")
  train$AllClusterID <- call$membership[train$itemID_1]
  
  # clusteroccurs
  aux <- aggregate(train$AllClusterID,by=list(train$AllClusterID),length)
  names(aux) <- c("AllClusterID","clusteroccurs")
  train <- merge(train,aux,by="AllClusterID",all.x=T)
  
  # clustersize 
  aux1 <- train[,c("AllClusterID","itemID_1")]
  aux2 <- train[,c("AllClusterID","itemID_2")]
  names(aux1) <- c("AllClusterID","itemID")
  names(aux2) <- c("AllClusterID","itemID")
  aux <- rbind(aux1,aux2)
  aux <- unique(aux)
  aux <- aggregate(aux$itemID,by=list(aux$AllClusterID),length)
  names(aux) <- c("AllClusterID","clustersize")
  train <- merge(train,aux,by="AllClusterID",all.x=T)
  
  # clustercover
  train$clustercover <- train$clusteroccurs*2 / (train$clustersize * (train$clustersize-1))
  write.csv(train[,c("itemID_1","itemID_2","clustersize","clusteroccurs","clustercover")],
            file=CLUSTER_TEST,row.names = F)
  
}

if (trainOrTest == "train") {
	print("Starting for TRAIN")

	pairs <- read_csv(train_ItemPairs)
	info <-  read_csv(train_ItemInfo)

	# Produced by: avito_word.py
	worddf1 <- read_feather(paste(cache_loc, "/features_train_set3c.fthr", sep = "" ))

	# Produced by: avito_json1.py
	worddf2 <- read_feather(paste(cache_loc, "/features_train_set3d.fthr", sep = "" ))

	# created by: images_train.py
	hamming <- read_feather(paste(cache_loc, "/features_train_set3f.fthr", sep = "" ))

	# Created by: avito_title.py
	title <- read_feather(paste(cache_loc, "/features_train_set3b.fthr", sep = "" ))

	# Created by: avito_description.py
	description <- read_feather(paste(cache_loc, "/features_train_set3a.fthr", sep = "" ))

	# Produced by: 
	new_json <- read_feather(paste(cache_loc, "/new_json_train_v1.fthr", sep = "" ))

	# created above
	cluster <- read_csv(CLUSTER_TRAIN)
	pairs <- merge(pairs,cluster,by=c("itemID_1","itemID_2"),all.x=T)
               
        source("code/3_feature_set3z_consolidate_internal.R")

	rm(info,worddf2,worddf1,title,hamming,description)
	gc()
	train <- pairs
	rm(pairs)
	gc()


	# Bow contains values for test and train
	# created with:  
	bow <- read_feather(paste(cache_loc, "/bow_models_train.fthr", sep = "" ))
	bow$isDuplicate <- NULL
	bow$valsample <- NULL
	train <- merge(train,bow,by=c("itemID_1","itemID_2"),all.x=T)

	# Save Data 
	train[is.na(train)]<- -1
	test[is.na(test)] <- -1 
	test$Var.145 <- NULL
	train$Var.146 <- NULL
	train$miss <- NULL
	test$miss <- NULL

	features <- setdiff(names(train),c("id", "itemID_1","itemID_2","isDuplicate","generationMethod","valsample"))

	label <- ifelse(train$generationMethod ==3,1,0)
	dtrain <- xgb.DMatrix(data=data.matrix(train[,features]),label=label) #temporary SOlution
	watchlist = list(train=dtrain)

	#Build Model to Predict Generation Method
	MODEL_GM3 <- paste(cache_loc, "/clf_gm3prob.RData", sep = "")
	set.seed(100)
	clf_gm3prob <- xgb.train(
	  data        = dtrain,
	  watchlist   = watchlist,
	  maximize    = T,
	  nrounds     = 100,
	  verbose     = 1,
	  print.every.n = 10,
	  early.stop.round = 100,
	  gamma = 5.0,
	  params      =  list(
		    objective = "binary:logistic",
		    booster = "gbtree",
		    eta = 0.1,
		    max_depth =12,
		    subsample = 0.6,
		    colsample_bytree = 0.6 ),
	            eval_metric = 'auc')

	save(clf_gm3prob,file=MODEL_GM3)

	# Predict gm3prob
	# ---------------
	dtrain <- xgb.DMatrix(data=data.matrix(train[,features]))
	gm3probV2 <- predict(clf_gm3prob,newdata=dtrain)
	rm(dtrain)
	train$gm3probV2 <- gm3probV2

	names(train) <- gsub("../code/", "", names(train))
	for (i in c("isDuplicate", "generationMethod", "id")) {
		train[[i]] <- NULL
	}
	names(train)[-(1:2)] <- paste("set3", names(train)[-(1:2)], sep = "_")

	print("Saving ")
	write_feather(train, paste(cache_loc, "/features_train_set3_consolidated.fthr", sep = ""))

print("done")
}

###################################################################
if (trainOrTest == "test") {
	print("Starting for TEST")

        pairs <- read_csv(test_ItemPairs)
        info <-  read_csv(test_ItemInfo)

        # Produced by: avito_word.py
        worddf1 <- read_feather(paste(cache_loc, "/features_test_set3c.fthr", sep = "" ))

        # Produced by: avito_json1.py
        worddf2 <- read_feather(paste(cache_loc, "/features_test_set3d.fthr", sep = "" ))

        # created by: images_test.py
        hamming <- read_feather(paste(cache_loc, "/features_test_set3f.fthr", sep = "" ))

        # Created by: avito_title.py
        title <- read_feather(paste(cache_loc, "/features_test_set3b.fthr", sep = "" ))

        # Created by: avito_description.py
        description <- read_feather(paste(cache_loc, "/features_test_set3a.fthr", sep = "" ))

        # Produced by:
        new_json <- read_feather(paste(cache_loc, "/new_json_test_v1.fthr", sep = "" ))

        # created above
        cluster <- read_csv(CLUSTER_TEST)
        pairs <- merge(pairs,cluster,by=c("itemID_1","itemID_2"),all.x=T)

        source("code/3_feature_set3z_consolidate_internal.R")

        rm(info,worddf2,worddf1,title,hamming,description)
        gc()
        test <- pairs
        rm(pairs)
        gc()


        # Bow contains values for test and test
        # created with:
        bow <- read_feather(paste(cache_loc, "/bow_models_test.fthr", sep = "" ))
        bow$isDuplicate <- NULL
        bow$valsample <- NULL
        test <- merge(test,bow,by=c("itemID_1","itemID_2"),all.x=T)

        # Save Data
        test[is.na(test)] <- -1
        test$Var.145 <- NULL
        test$miss <- NULL

        features <- setdiff(names(test),c("id", "itemID_1","itemID_2","isDuplicate","generationMethod","valsample"))


        MODEL_GM3 <- paste(cache_loc, "/clf_gm3prob.RData", sep = "")
	load(MODEL_GM3)

        # Predict gm3prob
        # ---------------
        dtest <- xgb.DMatrix(data=data.matrix(test[,features]))
        gm3probV2 <- predict(clf_gm3prob,newdata=dtest)
        rm(dtest)
        test$gm3probV2 <- gm3probV2

        names(test) <- gsub("../code/", "", names(test))
        for (i in c("isDuplicate", "generationMethod", "id")) {
                test[[i]] <- NULL
        }
        names(test)[-(1:2)] <- paste("set3", names(test)[-(1:2)], sep = "_")

        print("Saving ")
        write_feather(test, paste(cache_loc, "/features_test_set3_consolidated.fthr", sep = ""))

print("done")
} 

	
