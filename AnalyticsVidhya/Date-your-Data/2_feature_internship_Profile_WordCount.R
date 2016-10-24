library(dplyr)
library(tidyr)
library(readr)
library(stylo)
library(stringr)

#LOAD DATA
internship <- read_csv("../data/Internship_Processed.csv", na = c("", "NA", "NULL"))

########
getNGrams <- function(my.text, n = 1) {
  # which can be split into a vector of consecutive words:
  #my.vector.of.words = txt.to.words(my.text) #Removed this single is would replace all numbers
  #my.vector.of.words <- unlist(strsplit(gsub("\\s+", " ", str_trim(my.text)), " "))
  my.vector.of.words <- unlist(strsplit(gsub("\\s+", " ", my.text), " "))
  #my.vector.of.words <- unlist(strsplit(my.text, " "))
  # now, we create a vector of word 2-grams:
  if (length(my.vector.of.words) >= n) {
    make.ngrams(my.vector.of.words, ngram.size = n)
  } else {
    return(NULL)  
  }
}

###################################
getNgramsCount <- function(words, n) {
  #######################################
  # COUNTING NGRAMS FEATURES
  #######################################
  #Generate Ngrams
  NgramsProfile <- getNGrams(words, n)
  
  #Count of Ngrams
  countOfNgramsInProfile <- length(NgramsProfile)
  
  #Count of Unique NGrams
  countOfUniqueNgramsInProfile <- length(unique(NgramsProfile))
  
  return(c(countOfNgramsInProfile, countOfUniqueNgramsInProfile))
}

NCOL <- ncol(internship)
for ( n in 1:2) {
  print(n)
  internship_words <- as.data.frame(t(mapply(getNgramsCount, internship$Internship_Profile, n)))
  colnames(internship_words) <- c(paste("countOf_", n, "_gramsInProfile", sep = ""),
                              paste("countOfUnique_", n, "_gramsInProfile", sep = "")
                              )
  row.names(internship_words) <- NULL
  internship <- cbind(internship,  internship_words) 
}


write.csv(internship[, (NCOL+1):ncol(internship)], "../data/Features_internship_Profile_WordCount.csv", row.names = F)

