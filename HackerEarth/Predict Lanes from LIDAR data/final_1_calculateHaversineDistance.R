library(tidyverse)
library(stringr)
library(geosphere)

label <- read_csv("../input/labels.csv")
label$roadCoordinates <- NULL
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")
df_all <- bind_rows(train, test)

getDis <- function(x) {
  x <- as.data.frame(matrix(as.numeric(unlist(strsplit(unlist(strsplit(x, "\\|")), " "))), byrow = T, ncol = 2))
  x$V1 <- ifelse(x$V1 < -90, -90, x$V1)
  x$V2 <- ifelse(x$V2 < -90, -90, x$V2)
  x$V1 <- ifelse(x$V1 > 90, 90, x$V1)
  x$V2 <- ifelse(x$V2 > 90, 90, x$V2)
  x <- arrange(x, V1, V2)[c(1, nrow(x)), ]
  distHaversine(x[, 1], x[, 2])
}


getHaversineDistance <- function(id) {
  median(sapply(df_all$laneLineCoordinates[df_all$roadId == id], getDis, USE.NAMES = F)  )
}

roads <- data_frame(roadId = unique(df_all$roadId))
roads$haversineDistance <- (sapply(roads$roadId, getHaversineDistance))

write_csv(roads, "../input/roadsDistance.csv")


