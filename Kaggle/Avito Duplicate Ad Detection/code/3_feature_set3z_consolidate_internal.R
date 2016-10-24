# Read additional infos
location <- read_csv(location_csv)
category <- read_csv(category_csv)

# Merge JSON Frequency
info <- data.table(info)
setkey(info,attrsJSON)
info <- merge(info,jfreq,by="attrsJSON",all.x=T)
setkey(info,title)
info <- merge(info,tfreq,by="title",all.x=T)
setkey(info,description)
info <- merge(info,dfreq,by="description",all.x=T)

fjson <- function(i){
  if (is.na(info$attrsJSON[i])){
    NULL
  } else {
    fromJSON(info$attrsJSON[i],unexpected.escape="keep")
  }
}

icount <- function(x){
  length(strsplit(x,",")[[1]])
}

# Generate Features
info <- merge(info,location,by=c("locationID"),all.x=T)
info <- merge(info,category,by=c("categoryID"),all.x=T)

# Remove NA from text fields
info$title[is.na(info$title)] <- ""
info$description[is.na(info$description)] <- ""
info$attrsJSON[is.na(info$attrsJSON)] <- ""

# Some length informations
info$titlelength <- nchar(info$title)
info$desclength <- nchar(info$description)
info$jsonlength <- nchar(info$attrsJSON)
info$imagecount <- as.vector(sapply(info$images_array,icount))
info$imagecount[is.na(info$images_array)] <- 0

# Merge pairs with info
pairs <- merge(pairs,info,by.x=c("itemID_1"),by.y="itemID",all.x=T)
pairs <- merge(pairs,info,by.x=c("itemID_2"),by.y="itemID",all.x=T)

# check for exactly the same title, desc and json
pairs$sametitle <- ifelse(pairs$title.x==pairs$title.y,1,0)
pairs$samedesc <- ifelse(pairs$description.x==pairs$description.y,1,0)
pairs$samejson <- ifelse(pairs$attrsJSON.x==pairs$attrsJSON.y,1,0)



# relative price difference
pairs$pricediff <- 2.0*abs(pairs$price.x-pairs$price.y)/(pairs$price.x+pairs$price.y)
pairs$pricediff[is.na(pairs$price.x)] <- -1
pairs$pricediff[is.na(pairs$price.y)] <- -1
pairs$pricediff[is.na(pairs$price.x) & is.na(pairs$price.y)] <- -2

# absolut price difference
pairs$abspricediff <- abs(pairs$price.x-pairs$price.y)
pairs$abspricediff[is.na(pairs$price.x)] <- -1
pairs$abspricediff[is.na(pairs$price.y)] <- -1
pairs$abspricediff[is.na(pairs$price.x) & is.na(pairs$price.y)] <- -2

# mean price
pairs$meanprice <- 0.5*(pairs$price.x+pairs$price.y)
pairs$meanprice[is.na(pairs$price.y)] <- pairs$price.x[is.na(pairs$price.y)]
pairs$meanprice[is.na(pairs$price.x)] <- pairs$price.y[is.na(pairs$price.x)]
pairs$meanprice[is.na(pairs$price.y) & is.na(pairs$price.y)] <- -1

# Same region (Missing -1)
pairs$sameregion <- ifelse(pairs$regionID.x==pairs$regionID.y,1,0)
pairs$sameregion[is.na(pairs$sameregion)] <- -1
pairs$sameregion[is.na(pairs$regionID.x) & is.na(pairs$regionID.y)] <- -2

# Same metro (one is Missing: -1, both are missing: -2)
pairs$samemetro<- ifelse(pairs$metroID.x==pairs$metroID.y,1,0)
pairs$samemetro[is.na(pairs$samemetro)] <- -1
pairs$samemetro[is.na(pairs$metroID.x) & is.na(pairs$metroID.y)] <- -2

# Same location (one is Missing: -1, both are missing: -2)
pairs$samelocation <- ifelse(pairs$locationID.x==pairs$locationID.y,1,0)
pairs$samelocation[is.na(pairs$samelocation)] <- 1
pairs$samelocation[is.na(pairs$locationID.x) & is.na(pairs$locationID.y)] <- -2

# Distance between Locations
pairs$distance <- pointDistance(cbind(pairs$lon.x,pairs$lat.x),
                                cbind(pairs$lon.y,pairs$lat.y),lonlat=T)
pairs$logdistance <- as.integer(log10(pairs$distance+1))

# Difference between title length
pairs$reltitle <- 2.0*abs(pairs$titlelength.x-pairs$titlelength.y)/
  (pairs$titlelength.x+pairs$titlelength.y)
pairs$reltitle[pairs$titlelength.x==0] <- -1 
pairs$reltitle[pairs$titlelength.y==0] <- -1 
pairs$reltitle[(pairs$titlelength.y+pairs$titlelength.y)==0] <- -2 

# relative difference in title length
pairs$reldesc <- 2.0*abs(pairs$desclength.x-pairs$desclength.y)/
  (pairs$desclength.x+pairs$desclength.y)
pairs$reldesc[pairs$desclength.x==0] <- -1  
pairs$reldesc[pairs$desclength.y==0] <- -1 
pairs$reldesc[(pairs$desclength.x+pairs$desclength.y)==0] <- -2  

# relative difference in json length
pairs$reljson <- 2.0*abs(pairs$jsonlength.x-pairs$jsonlength.y)/(pairs$jsonlength.x+pairs$jsonlength.y)
pairs$reljson[pairs$jsonlength.x==0] <- -1 
pairs$reljson[pairs$jsonlength.y==0] <- -1 
pairs$reljson[pairs$jsonlength.y+pairs$jsonlength.y==0] <- -2

# imagescounts
pairs$diffimagecount <- abs(pairs$imagecount.x-pairs$imagecount.y)
pairs$maximagescount <- pmax(pairs$imagecount.x,pairs$imagecount.y)
pairs$minimagescount <- pmin(pairs$imagecount.x,pairs$imagecount.y)

# min an max length 
pairs$maxtitlelength <- pmax(pairs$titlelength.x,pairs$titlelength.y)
pairs$mintitlelength <- pmin(pairs$titlelength.x,pairs$titlelength.y)
pairs$maxdesclength <- pmax(pairs$desclength.x,pairs$desclength.y)
pairs$mindesclength <- pmin(pairs$desclength.x,pairs$desclength.y)
pairs$maxjsonlength <- pmax(pairs$jsonlength.x,pairs$jsonlength.y)
pairs$minjsonlength <- pmin(pairs$jsonlength.x,pairs$jsonlength.y)
pairs$desclength.x <-NULL
pairs$desclength.y <- NULL
pairs$titlelength.x <- NULL
pairs$titlelength.y <- NULL
pairs$jsonlength.x <- NULL
pairs$jsonlength.y <- NULL

pairs$deltalocation <- abs(pairs$locationID.x-pairs$locationID.y)
pairs$deltalocation[is.na(pairs$deltalocation)] <- -1
pairs$metroID <- ifelse(pairs$metroID.y==pairs$metroID.x,pairs$metroID.x,-1)
pairs$metroID[is.na(pairs$metroID)] <- -1
pairs$regionID <- ifelse(pairs$regionID.x==pairs$regionID.y,pairs$regionID.x,-1)
pairs$regionID[is.na(pairs$regionID)] <- -1 

# Remove redundacies
pairs$categoryID = pairs$categoryID.x
pairs$categoryID.x <- NULL
pairs$categoryID.y <- NULL
pairs$parentCategoryID <- pairs$parentCategoryID.x
pairs$parentCategoryID.x <- NULL
pairs$parentCategoryID.y <- NULL
pairs$lat.x <-NULL
pairs$lat.y <-NULL
pairs$lon.x <- NULL
pairs$lon.y <- NULL
pairs$imagecount.x <- NULL
pairs$imagecount.y <- NULL
pairs$metroID.x <- NULL
pairs$metroID.y <- NULL
pairs$regionID.x <- NULL
pairs$regionID.y <- NULL
#pairs$locationID.x <- NULL
#pairs$locationID.y <- NULL
pairs$price.x <- NULL
pairs$price.y <- NULL
pairs$images_array.x <- NULL
pairs$images_array.y <- NULL
pairs$title.x <- NULL
pairs$title.y <- NULL
pairs$description.x <- NULL
pairs$description.y <- NULL
pairs$attrsJSON.x <- NULL
pairs$attrsJSON.y <- NULL

worddf1 <- worddf1[,c("itemID_1","itemID_2","simjson","matjson1","matjson2")]
worddf1$minmatchjson <- pmin(worddf1$matjson1,worddf1$matjson2)
worddf1$maxmatchjson <- pmax(worddf1$matjson1,worddf1$matjson2)
worddf1$matjson1 <- NULL
worddf1$matjson2 <- NULL
pairs <- merge(pairs,worddf1,by=c("itemID_1","itemID_2"),all.x=T)

pairs <- merge(pairs,worddf2,by=c("itemID_1","itemID_2"),all.x=T)
pairs$d_similarkeys[is.na(pairs$similarkeys)] <- -1
pairs$d_similarvals[is.na(pairs$similarvals)] <- -1
pairs$d_nkey1[is.na(pairs$nkey1)] <- -1
pairs$d_nkey2[is.na(pairs$nkey2)] <- -1
pairs$diffkey <- abs(pairs$d_nkey1-pairs$nkey2)
pairs$maxkey <- pmax(pairs$d_nkey1,pairs$nkey2)
pairs$minkey <- pmin(pairs$d_nkey1,pairs$nkey2)
pairs$relkey = pairs$diffkey / pairs$maxkey
pairs$relkey[is.na(pairs$relkey)] <- -1
pairs$d_nkey1 <- NULL
pairs$d_nkey2 <- NULL


# merge image comparison
hamming$ham1 <- hamming$ham1 + hamming$ham0
hamming$ham2 <- hamming$ham2 + hamming$ham1
hamming$ham3 <- hamming$ham3 + hamming$ham2
hamming$ham4 <- hamming$ham4 + hamming$ham3
hamming$ham5 <- hamming$ham5 + hamming$ham4
hamming$ham6 <- hamming$ham6 + hamming$ham5
hamming$ham7 <- hamming$ham7 + hamming$ham6
hamming$ham8 <- hamming$ham7 + hamming$ham7
pairs <- merge(pairs,hamming,by=c("itemID_1","itemID_2"),all.x=T)

pairs$identimg <- pairs$ham0
pairs$identimg[is.na(pairs$identimg)] <- -1
pairs$relidentimage <- pairs$identimg / pairs$maximagescount
pairs$relidentimage[is.na(pairs$relidentimage)] <- -1
pairs$relidentimage[pairs$maximagescount==0] <- -1

pairs$identimg8 <- pairs$ham8
pairs$identimg8[is.na(pairs$identimg8)] <- -1
pairs$relidentimage8 <- pairs$identimg8 / pairs$maximagescount
pairs$relidentimage8[is.na(pairs$relidentimage8)] <- -1
pairs$relidentimage8[pairs$maximagescount==0] <- -1
pairs$minidentimage8 <- pairs$identimg8 / pairs$minimagescount
pairs$minidentimage8[is.na(pairs$minidentimage8)] <- -1
pairs$minidentimage8[pairs$minimagescount==0] <- -1
h1 <- paste("ham",0:8,sep="")
for (i in h1){ pairs[,i]<-NULL}
pairs$maxoccur <- pmax(pairs$maxnx,pairs$maxny)
pairs$minoccur <- pmin(pairs$maxnx,pairs$maxny)
pairs$minoccur[is.na(pairs$minoccur)] <- -1
pairs$maxoccur[is.na(pairs$maxoccur)] <- -1
pairs$maxnx <- NULL
pairs$maxny <- NULL
pairs$miss[is.na(pairs$miss)] <- 0
pairs$minham[is.na(pairs$minham)] <- -1
pairs$minham100[is.na(pairs$minham100)] <- -1.0
pairs$minham50[is.na(pairs$minham50)] <- -1.0
pairs$minham30[is.na(pairs$minham30)] <- -1.0
pairs$hasimages <- ifelse(pairs$maximagescount>0,1,0) + ifelse(pairs$minimagescount>0,1,0)


pairs$maxjsonFreq <- pmax(pairs$jsonFreq.x,pairs$jsonFreq.y)
pairs$minjsonFreq <- pmin(pairs$jsonFreq.x,pairs$jsonFreq.y)
pairs$maxjsonFreq <- as.numeric(ceiling(log(pairs$maxjsonFreq)))
pairs$minjsonFreq <- as.numeric(ceiling(log(pairs$minjsonFreq)))
pairs$jsonFreq.x <- NULL
pairs$jsonFreq.y <- NULL

pairs$maxtitleFreq <- pmax(pairs$titleFreq.x,pairs$titleFreq.y)
pairs$mintitleFreq <- pmin(pairs$titleFreq.x,pairs$titleFreq.y)
pairs$maxtitleFreq <- as.numeric(ceiling(log(pairs$maxtitleFreq)))
pairs$mintitleFreq <- as.numeric(ceiling(log(pairs$mintitleFreq)))
pairs$titleFreq.x <- NULL
pairs$titleFreq.y <- NULL

pairs$maxdescFreq <- pmax(pairs$descFreq.x,pairs$descFreq.y)
pairs$mindescFreq <- pmin(pairs$descFreq.x,pairs$descFreq.y)
pairs$maxdescFreq <- as.numeric(ceiling(log(pairs$maxdescFreq)))
pairs$mindescFreq <- as.numeric(ceiling(log(pairs$mindescFreq)))
pairs$descFreq.x <- NULL
pairs$descFreq.y <- NULL

# New Title festures
colnames(title)<-c("itemID_1","itemID_2","new_simtitle","new_mattitle1","new_mattitle2","titlenwords1","titlenwords2")
pairs <- merge(pairs,title,by=c("itemID_1","itemID_2"),all.x=T)
pairs$maxmatchtitle <- pmax(pairs$new_mattitle1,pairs$new_mattitle2)
pairs$minmatchtitle <- pmin(pairs$new_mattitle1,pairs$new_mattitle2)
pairs$mattitle1 <- NULL
pairs$mattitle2 <- NULL
pairs$maxtitlewords <- pmax(pairs$titlenwords1,pairs$titlenwords2)
pairs$mintitlewords <- pmin(pairs$titlenwords1,pairs$titlenwords2)
pairs$difftitlewords <- abs(pairs$titlenwords1-pairs$titlenwords2)
pairs$titlenwords1 <-NULL
pairs$titlenwords2 <- NULL

description$minmatchdesc <- pmin(description$mat1_d,description$mat2_d)
description$maxmatchdesc <- pmax(description$mat1_d,description$mat2_d)
description$maxdescwords <- pmax(description$nwords1,description$nwords2)
description$maxdescwords <- pmin(description$nwords1,description$nwords2)
description$diffdescwords <- abs(description$nwords1-description$nwords2)
description$new_simdesc <- description$simdesc
description$simdesc <- NULL
description$matdesc1 <- NULL
description$matdesc2 <- NULL
description$nwords1 <- NULL
description$nwords2 <- NULL
pairs <- merge(pairs,description,by=c("itemID_1","itemID_2"),all.x=T)

# pairs <- merge(pairs,gm3[,c("itemID_1","itemID_2","generation3prob")],by=c("itemID_1","itemID_2"),all.x=T)

#pairs$generation3prob <- round(pairs$generation3prob)
pairs$logmeanprice <- round(log(pairs$meanprice+1.),1)
pairs$pricediff <- round(pairs$pricediff,3)
pairs$reldesc <- round(pairs$reldesc,3)
pairs$reljson <- round(pairs$reljson,3)
pairs$reltitle <- round(pairs$reltitle,3)
pairs$logabspricediff <- round(log(pairs$abspricediff+1),1)
pairs$logabspricediff[is.na(pairs$logabspricediff)] <- -1
price_diff <- function(x,delta)
{
  ifelse(pmin(x %% delta,-x %% delta)<2,1,0) 
}
pairs$pricediff100 <- price_diff(pairs$abspricediff,100)
pairs$pricediff500 <- price_diff(pairs$abspricediff,500)
pairs$pricediff1000 <- price_diff(pairs$abspricediff,1000)
pairs$pricediff5000 <- price_diff(pairs$abspricediff,5000)

aux <- dummy(pairs$parentCategoryID)
pairs <- cbind(pairs,aux)
#pairs$parentCategoryID <- NULL
names(pairs) <- gsub("code/3_feature_set3z_consolidate_internal.R","parentCategoryID",names(pairs))

aux <- dummy(pairs$categoryID)
pairs <- cbind(pairs,aux)
#pairs$parentCategoryID <- NULL
names(pairs) <- gsub("code/3_feature_set3z_consolidate_internal.R","categoryID",names(pairs))

# Bind new json features
pairs <- merge(pairs,new_json,by=c("itemID_1","itemID_2"),all.x=T)

# Add Bricks features
#pairs <- merge(pairs,bricks,by=c("itemID_1","itemID_2"),all.x=T)


# make all features numeric
for (i in 3:length(pairs)){
  pairs[,i] <- as.numeric(pairs[,i])
}







  
