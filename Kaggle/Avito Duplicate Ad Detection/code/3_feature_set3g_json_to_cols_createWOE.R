# Create new jsons features
library(readr)
library(Hmisc)
library(data.table)
library(reshape2)
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
JSON_WOE_MODE <- paste(cache_loc, "/woe.Rdata", sep = "")


if (!file.exists(JSON_WOE_MODE)) {
	print("Generating Json Woe")
        json <- read_feather(paste(cache_loc, "json_vals_train_v2.fthr", sep = ""))
        json$value <- as.numeric(json$value)

        json$value[json$value>0.8] <-1
        json$value[json$value<1 & json$value> -1] <-0

        json <- data.frame(json)
        json <- dcast(json,itemID_1+itemID_2~keyID)

        pairs <- read_csv(train_ItemPairs)
        setDT(pairs)
        setDT(json)
        setkey(pairs,itemID_1,itemID_2)
        setkey(json,itemID_1,itemID_2)
	json <- merge(pairs,json,by=c("itemID_1","itemID_2"),all.x=T)

        json <- data.frame(json)
        for (i in colnames(json)){
          json[is.na(json[,i]),i] <- -2
        }

        gc()
        rm(pairs)

        # calculate pairwise weights of evidence

        #  0  1
        #######
        woe2 <- data.frame()
        k<-0
        totdup <- sum(json$isDuplicate)
        totnodup <- nrow(json) -sum(json$isDuplicate)
        for (i in 5:183){
          n1 <- names(json)[i]
          print(i)
          for (j in 5:183){
            n2 <- names(json)[j]
            f <- json[,i]<0 & json[,j]==1
            aux <- json$isDuplicate[f]
            if (length(aux)>0){
              k <- k+1
              woe2[k,"json1"] <- n1
              woe2[k,"json2"] <- n2
              ndup <- sum(aux)
              nodup <- length(aux)-ndup
              woe2[k,"count"] <-ndup +nodup
              woe2[k,"woe"] <- log(ndup/totdup / (nodup/totnodup))
              #print(woe2[k,])
            }
          }
        }

        woe1 <- woe2[woe2$json1==woe2$json2,]
        woe1$json2 <- NULL
        colnames(woe1)[1] <- "json"

        woe2_01 <- woe2
        woe1_01 <- woe1


        #  1  1
        #######
        woe2 <- data.frame()
        k<-0
        totdup <- sum(json$isDuplicate)
        totnodup <- nrow(json) -sum(json$isDuplicate)
        for (i in 5:183){
          print(i)
          n1 <- names(json)[i]
          for (j in 5:183){
            n2 <- names(json)[j]
            f <- json[,i]==1 & json[,j]==1
            aux <- json$isDuplicate[f]
            if (length(aux)>0){
              k <- k+1
              woe2[k,"json1"] <- n1
              woe2[k,"json2"] <- n2
              woe2[k,"count"] <- length(aux)
              woe2[k,"dups"] <- sum(aux)
              woe2[k,"meandup"] <- mean(aux)
              ndup <- sum(aux)
              nodup <- woe2[k,"count"]-ndup
              woe2[k,"woe"] <- log(ndup/totdup / (nodup/totnodup))
              woe2[k,"woe_neg"] <- log((totdup-ndup)/totdup / ((totnodup-nodup)/totnodup))
              #print(woe2[k,])
            }
          }
        }

        woe1 <- woe2[woe2$json1==woe2$json2,]
        woe1$json2 <- NULL
        colnames(woe1)[1] <- "json"

        woe2_11 <- woe2
        woe1_11 <- woe1

        #  0  0
        #######
        woe2 <- data.frame()
        k<-0
        totdup <- sum(json$isDuplicate)
        totnodup <- nrow(json) -sum(json$isDuplicate)
        for (i in 5:183){
          print(i)
          n1 <- names(json)[i]
          for (j in 5:183){
            n2 <- names(json)[j]
            f <- json[,i]==0 & json[,j]==0.0
            aux <- json$isDuplicate[f]
            aux2 <- json$PRED[f]
            if (length(aux)>0){
              k <- k+1
              woe2[k,"json1"] <- n1
              woe2[k,"json2"] <- n2
              woe2[k,"count"] <- length(aux)
              woe2[k,"dups"] <- sum(aux)
              woe2[k,"meandup"] <- mean(aux)
              ndup <- sum(aux)
              nodup <- woe2[k,"count"]-ndup
              woe2[k,"woe"] <- log(ndup/totdup / (nodup/totnodup))
              woe2[k,"woe_neg"] <- log((totdup-ndup)/totdup / ((totnodup-nodup)/totnodup))
        #      print(woe2[k,])
            }
          }
        }

        woe1 <- woe2[woe2$json1==woe2$json2,]
        woe1$json2 <- NULL
        colnames(woe1)[1] <- "json"

        woe2_00 <- woe2
        woe1_00 <- woe1

        save.image(file = JSON_WOE_MODE)

} 

load(JSON_WOE_MODE)
#rm(list = setdiff(ls(), c(grep("woe", ls(), value = T), trainOrTest)))

calc_new_json <- function(){
  json <-data.table(json)
  jsonentries<- json[,.(.N),by=list(itemID_1,itemID_2)]
  colnames(jsonentries) <- c("itemID_1","itemID_2","jsonentries")
  jsonequal <- json[json$value==1,.(.N),by=list(itemID_1,itemID_2)]
  colnames(jsonequal) <- c("itemID_1","itemID_2","jsonequal")
  jsonnotequal <- json[json$value==0,.(.N),by=list(itemID_1,itemID_2)]
  colnames(jsonnotequal) <- c("itemID_1","itemID_2","jsonnotequal")
  jsonnopairs <- json[json$value==-1,.(.N),by=list(itemID_1,itemID_2)]
  colnames(jsonnopairs) <- c("itemID_1","itemID_2","jsonnopairs")
  jsontotsim <- json[json$value>0,sum(value),by=list(itemID_1,itemID_2)]
  colnames(jsontotsim) <- c("itemID_1","itemID_2","jsontotsim")
  json[is.na(json)] <- -1
  json <- data.frame(json)
  json <- dcast(json,itemID_1+itemID_2~keyID)
  json <- merge(pairs,json,by=c("itemID_1","itemID_2"),all.x=T)
  json[is.na(json)] <- -2
  json <- data.frame(json)
  json <- merge(json,jsonequal,by=c("itemID_1","itemID_2"),all.x=T)
  json <- merge(json,jsonnotequal,by=c("itemID_1","itemID_2"),all.x=T)
  json <- merge(json,jsonentries,by=c("itemID_1","itemID_2"),all.x=T)
  json <- merge(json,jsonnopairs,by=c("itemID_1","itemID_2"),all.x=T)
  json <- merge(json,jsontotsim,by=c("itemID_1","itemID_2"),all.x=T)
  json <- data.frame(json)
  json$jsonequal[is.na(json$jsonequal)] <- -1
  json$jsonentries[is.na(json$jsonentries)] <- -1
  json$jsonnotequal[is.na(json$jsonnotequal)] <- -1
  json$jsonnopairs[is.na(json$jsonnopairs)] <- -1
  json$jsontotsim[is.na(json$jsontotsim)] <- -1
  json$jsonrelequal  <- json$jsonequal / json$jsonnopairs
  json$jsonrelequal[json$jsonequal == -1 | json$jsonpairs == -1] <- -1
  
  jsonnames <- names(json)
  
  # calculate 1 grams
  json$woe1_0 <- 0.0
  for (i in 1:nrow(woe1_00)){
    print(i)
    if (woe1_00$count[i]>20){
      j <- woe1_00$json[i]
      woe <- woe1_00$woe[i]
      woe <- sign(woe) * min(abs(woe),5.0)
      if (j %in% jsonnames) {
      	aux <- ifelse(json[,j]==0,1,0)
      	json$woe1_0 <- json$woe1_0 +  woe * aux
      }
    }
  }
  
  json$woe1_1 <- 0.0
  for (i in 1:nrow(woe1_00)){
    print(i)
    if (woe1_11$count[i]>20){
      j <- woe1_11$json[i]
      woe <- woe1_11$woe[i]
      woe <- sign(woe) * min(abs(woe),5.0)
      if (j %in% jsonnames) {
      	aux <- ifelse(json[,j]==1,1,0)
      	json$woe1_1 <- json$woe1_1 +  woe * aux
      }
    }
  }
  
  # calculate woe on 2 grams 
  json$woe2_00 <- 0.0
  
  for (i in 1:nrow(woe2_00)){
    print(i)
    if (woe2_00$count[i] >20){
      json1 <- woe2_00$json1[i]
      json2 <- woe2_00$json2[i]
      if (json1 %in% jsonnames & json2 %in% jsonnames){
      	woe <- woe2_00$woe[i]
      	woe <- sign(woe) * min(abs(woe),5.0)
      	json$woe2_00 <- json$woe2_00 +  ifelse(json[,json1]==0 & json[,json2]==0,woe,0)
	}
    }
  }
  
  
  json$woe2_11 <- 0.0
  for (i in 1:nrow(woe2_11)){
    print(i)
    if (woe2_11$count[i]>20){
      json1 <- woe2_11$json1[i]
      json2 <- woe2_11$json2[i]
      if (json1 %in% jsonnames & json2 %in% jsonnames){
      	woe <- woe2_11$woe[i]
      	woe <- sign(woe) * min(abs(woe),5.0)
      	json$woe2_11 <- json$woe2_11 +  ifelse(json[,json1]==1 & json[,json2]==1,woe,0)
      }
    }
  }
  
  json$woe2_01 <- 0.0
  for (i in 1:nrow(woe2_01)){
    print(i)
    if (woe2_01$count[i]>20){
      json1 <- woe2_01$json1[i]
      json2 <- woe2_01$json2[i]
      if (json1 %in% jsonnames & json2 %in% jsonnames){
        woe <- woe2_01$woe[i]
        woe <- sign(woe) * min(abs(woe),5.0)
        json$woe2_01 <- json$woe2_01 +  ifelse(json[,json1]==0 & json[,json2]==1,woe,0)
      }
    }
  }
 return(json)
  
}
if (trainOrTest == "train") {
        # train data
	print("Generating WoE for TRAIN")
        pairs <- read_csv(train_ItemPairs)
        json <- read_feather(paste(cache_loc, "json_vals_train_v2.fthr", sep = ""))
        json$value <- as.numeric(json$value)
        json <- calc_new_json()

        new_json_features <- json[,c("itemID_1","itemID_2","woe2_00","woe2_11","woe2_01",
                                     "woe1_0","woe1_1","jsonequal","jsonnotequal",
                                     "jsonentries","jsonnopairs","jsontotsim")]
        write_feather(new_json_features,paste(cache_loc, "/json_vals_train_v2.fthr", sep = ""))
}

###################################################################
if (trainOrTest == "test") {
        # test data
	print("Generating WoE for TEST")
        pairs <- read_csv(test_ItemPairs)
        json <- read_feather(paste(cache_loc, "json_vals_test_v2.fthr", sep = ""))
        json$value <- as.numeric(json$value)
        json <- calc_new_json()

        new_json_features <- json[,c("itemID_1","itemID_2","woe2_00","woe2_11","woe2_01",
                                     "woe1_0","woe1_1","jsonequal","jsonnotequal",
                                     "jsonentries","jsonnopairs","jsontotsim")]
        write_feather(new_json_features,paste(cache_loc, "/json_vals_test_v2.fthr", sep = ""))
}


