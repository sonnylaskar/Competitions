#### Copyright 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Avito Duplicate Ad Detection
# functions.R
# TODO: WRITE DESCRIPTION OF SCRIPT HERE

#Load Basic packages needed by all R scripts
library(readr)
library(dplyr)
library(tidyr)
library(feather)

######## GET NGRAMS FUNCTIONS
getNGrams <- function(my.text, n = 1) {
  # which can be split into a vector of consecutive words:
  my.vector.of.words <- stemDocument(unlist(strsplit(gsub("\\s+", " ", str_trim(my.text)), " ")))
  # now, we create a vector of word n-grams:
  if (length(my.vector.of.words) >= n) {
    make.ngrams(my.vector.of.words, ngram.size = n)
  } else {
    return(NULL)
  }
}
######## GET NCHARS FUNCTIONS
getNGramsChars <- function(my.text, n = 1) {
  # which can be split into a vector of consecutive words:
  my.vector.of.words <- stemDocument(unlist(strsplit(gsub("\\s+", " ", str_trim(my.text)), " ")))
  # now, we create a vector of word n-grams:
  if (length(my.vector.of.words) >= n) {
    my.vector.of.chars = txt.to.features(my.vector.of.words, features = "c")
    make.ngrams(my.vector.of.chars, ngram.size = n)
  } else {
    return(NULL)
  }
}

## NGRAMS
getNgramsCount <- function(string1, string2, n = 1) {
  #######################################
  # COUNTING NGRAMS FEATURES
  #######################################
  #Generate Ngrams
  NgramsString1 <- getNGrams(tolower(string1), n)
  NgramsString2 <- getNGrams(tolower(string2), n)

  #Count of Ngrams
  countOfNgramsInString1 <- length(NgramsString1)
  countOfNgramsInString2 <- length(NgramsString2)
  ratioOfNgrams_String1_String2 <- round(countOfNgramsInString1 / countOfNgramsInString2, 3)

  #Count of Unique NGrams
  countOfUniqueNgramsInString1 <- length(unique(NgramsString1))
  countOfUniqueNgramsInString2 <- length(unique(NgramsString2))
  ratioOfUniqueNgrams_String1_String2 <- round(countOfUniqueNgramsInString1 / countOfUniqueNgramsInString2, 3)

  ratioOfIntersect_Ngrams_String1_in_String2 <- round(sum(NgramsString1 %in% NgramsString2) / countOfNgramsInString1, 3)
  ratioOfIntersect_Ngrams_String2_in_String1 <- round(sum(NgramsString2 %in% NgramsString1) / countOfNgramsInString2, 3)

  countOfNgramsInString_min <- min( countOfNgramsInString1, countOfNgramsInString2 )
  countOfNgramsInString_max <- max( countOfNgramsInString1, countOfNgramsInString2 )
  countOfNgramsInString_sum <- ( countOfNgramsInString1 + countOfNgramsInString2 )
  countOfNgramsInString_diff <- abs( countOfNgramsInString1 - countOfNgramsInString2 )

  return(c(
        countOfNgramsInString_min,
        countOfNgramsInString_max,
        countOfNgramsInString_sum,
        countOfNgramsInString_diff,
	countOfNgramsInString1,
	countOfNgramsInString2,
	countOfUniqueNgramsInString1,
	countOfUniqueNgramsInString2,
	ratioOfNgrams_String1_String2, 
	ratioOfUniqueNgrams_String1_String2,
	ratioOfIntersect_Ngrams_String1_in_String2,
	ratioOfIntersect_Ngrams_String2_in_String1
  	))
}

## NCHARS
getNcharsCount <- function(string1, string2, n = 1) {
  #######################################
  # COUNTING Nchars FEATURES
  #######################################
  #Generate Nchars
  NcharsString1 <- getNGramsChars(tolower(string1), n)
  NcharsString2 <- getNGramsChars(tolower(string2), n)

  #Count of Nchars
  countOfNcharsInString1 <- length(NcharsString1)
  countOfNcharsInString2 <- length(NcharsString2)
  ratioOfNchars_String1_String2 <- round(countOfNcharsInString1 / countOfNcharsInString2, 3)

  #Count of Unique Nchars
  countOfUniqueNcharsInString1 <- length(unique(NcharsString1))
  countOfUniqueNcharsInString2 <- length(unique(NcharsString2))
  ratioOfUniqueNchars_String1_String2 <- round(countOfUniqueNcharsInString1 / countOfUniqueNcharsInString2, 3)

  ratioOfIntersect_Nchars_String1_in_String2 <- round(sum(NcharsString1 %in% NcharsString2) / countOfNcharsInString1, 3)
  ratioOfIntersect_Nchars_String2_in_String1 <- round(sum(NcharsString2 %in% NcharsString1) / countOfNcharsInString2, 3)

  countOfNcharsInString_min <- min( countOfNcharsInString1, countOfNcharsInString2 )
  countOfNcharsInString_max <- max( countOfNcharsInString1, countOfNcharsInString2 )
  countOfNcharsInString_sum <- ( countOfNcharsInString1 + countOfNcharsInString2 )
  countOfNcharsInString_diff <- abs(( countOfNcharsInString1 - countOfNcharsInString2 ))

  return(c(
	countOfNcharsInString_min,
	countOfNcharsInString_max,
	countOfNcharsInString_sum,
	countOfNcharsInString_diff,
	countOfNcharsInString1,
	countOfNcharsInString2,
	countOfUniqueNcharsInString1,
	countOfUniqueNcharsInString2,
        ratioOfNchars_String1_String2,
        ratioOfUniqueNchars_String1_String2,
        ratioOfIntersect_Nchars_String1_in_String2,
        ratioOfIntersect_Nchars_String2_in_String1
        ))
}


