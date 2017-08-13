library(tidyverse)
library(feather)
library(stringr)

label <- read_csv("../input/labels.csv")
label$roadCoordinates <- NULL
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")
df_all <- bind_rows(train, test)
roadsDistance <- read_csv("../input/roadsDistance.csv")

getLatLong <- function(x, t = "lat") {
  a <- matrix(as.numeric(unlist(strsplit(unlist(strsplit(x, "\\|")), " "))), byrow = T, ncol = 2)
  if (t == "lon") { 
    apply(a, 2, mean)[1]
  } else {
    apply(a, 2, mean)[2]
  }
}


df_all$meanLat <- sapply(df_all$laneLineCoordinates, getLatLong, t = "lat", USE.NAMES = F)
df_all$meanLon <- sapply(df_all$laneLineCoordinates, getLatLong, t = "lon", USE.NAMES = F)  
  
df_all %>%
  group_by(roadId) %>%
  summarise(
    sumOfDistanceFromLeft = sum(distFromLaneLineOnLeft, na.rm = T),
    sumOfDistanceFromRight = sum(distFromLaneLineOnRight, na.rm = T),
    r_sumOfDistanceFromLR = sumOfDistanceFromLeft / sumOfDistanceFromRight,
    int_distLR = length(intersect(distFromLaneLineOnLeft, distFromLaneLineOnRight)),

    latCounter = length(unique(round(meanLat, 4))),
    lonCounter = length(unique(round(meanLon, 4))),

    int_TotalLinesLR = length(intersect(totalLinesOnLeft, totalLaneLinesOnRight)),
    uniq_linesLeft = length(unique(totalLinesOnLeft)),
    uniq_linesRight = length(unique(totalLaneLinesOnRight)),
    totalLaneLinesMean = mean(totalLaneLines),
    totalLinesOnLeftMax = max(totalLinesOnLeft),

    uniq_lineId = length(unique(laneLineId)) / length((laneLineId)),
    roadCategory = unique(roadCategory),

    r_lineToRoadLength = sum(laneLineLength / roadLength < 0.8),
    r_lineToRoadLength2 = sum(laneLineLength / roadLength >= 0.8),
    laneLineLengthMean = mean(laneLineLength),
    
    sum_interSectingLines = sum(noOfIntersectingLaneLinesLeft, noOfIntersectingLaneLinesRight),
    noOfIntersectingLaneLinesLeftMean = mean(noOfIntersectingLaneLinesLeft),

    sum_isIntersectingWithRoadGeometryTrue = sum(isIntersectingWithRoadGeometry == "true"),
    sum_isIntersectingWithRoadGeometryFalse = sum(isIntersectingWithRoadGeometry == "false")
  ) -> df2



df2$data <- ifelse(df2$roadId %in% train$roadId, "train", "test")
df2 <- left_join(df2, roadsDistance, by = "roadId")
df2$haversineDistance <- df2$haversineDistance / df2$laneLineLengthMean
df2 <- left_join(df2, label, by = "roadId")

write_feather(df2, "../input/df_all.fthr")

