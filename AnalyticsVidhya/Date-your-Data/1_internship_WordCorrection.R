library(qdap)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(tm)

#LOAD DATA
internship <- read_csv("../data/Internship.csv", na = c("", "NA", "NULL"))

SPELLINGERRORS <- check_spelling(internship$Internship_Profile,
                      assume.first.correct = TRUE,
                      n.suggests = 4)
SPELLINGERRORS <- data.frame(lapply(SPELLINGERRORS, as.character), 
                             stringsAsFactors=FALSE) %>%
                  select(not.found, suggestion)
#Remove Duplicate rows
SPELLINGERRORS <- SPELLINGERRORS[!duplicated(SPELLINGERRORS[1:2]), ]

#Now check sort(SPELLINGERRORS$not.found) and see which are actual spelling mistakes, which are correct but need modification
#Below are what I have observed:
SPELL_MISTAKES <- c("activites", "ambassodor", "andoid", "andorid", "andriod", "anubhava","autid","bussiness","chemsitry",
                    "coordinaing","cosnulting","develoment","developement","develpoment","enrolment","facilitation",
                    "finanace","managemnt","managment","mangement","marekting","markting","notejs","nutritionist","oflline","optimaization",
                    "optimization","optmization","pharmacovigilance","reasearch","recruiter","professonal","requirment","retreival","socia",
                    "trbology","tution","varification","vertification","writitng")

SPELLINGERRORS <- SPELLINGERRORS[(SPELLINGERRORS$not.found %in% SPELL_MISTAKES), ]
SIMILAR_WORDS <- list(
  c("apps", "app"), 
  c("Accounting", "Accountant"),
  c("back-end", "backend"),
  c("beckend", "backend"),
  c("back end", "backend"),
  c("blog", "blogger"),
  c("blogging", "blogger"),
  c("blogs", "blogger"),
  c("cataloguing" ,"catalogue"),
  c("curating", "curation"),
  c("desiging", "design"),
  c("desigining", "design"),
  c("designe", "design"),
  c("telecalling", "telecaller"),
  c("telecommunications", "telecom"),
  c("trbology" , "tribology"),
  c("oflline", "offline")
)
m <- matrix(unlist(SIMILAR_WORDS), byrow = TRUE, ncol = 2)
colnames(m) <- c("not.found", "suggestion")
SPELLINGERRORS <- rbind(SPELLINGERRORS, m)

#Function to replace Spelling errors
replaceSpellingErrors <- function(words) {
  b <- c()
  for (i in unlist(strsplit(words, " "))) {
    if (i %in% SPELLINGERRORS$not.found) {
      b <- append(b, SPELLINGERRORS$suggestion[SPELLINGERRORS$not.found == i])
    } else {
      b <- append(b, i)
    }
  }
  return(paste(b, collapse = " "))
}

#Function to remove all unwanted stuff
cleanUpText <- function(words, stem = TRUE) {
  #Remove all graph characters
  words <- str_replace_all(words,"[^[:graph:]]", " ")
  words <- gsub("[^[:alpha:][:space:]]*", "", words)
  words <- tolower(words)
  #Remove Punctuation except Hyphen -
  words <- gsub("([-])|[[:punct:]]", '\\1', words)
  #Remove all extra whitespace
  gsub("\\s+", " ", str_trim(words))
  #Replace all spelling errors
  words <- replaceSpellingErrors(words)
  #Stemming if stem = TRUE
  stemList <- c()
  if (stem) {
    for (i in words) {
      i <- gsub("[[:punct:]]$", "", i) #Remove any trailing punctuation mark
      i <- gsub("^[[:punct:]]", "", i) #Remove any leading punctuation mark
      j <- paste(stemDocument(unlist(strsplit(i," "))), collapse = " ")
      stemList <- append(stemList, j)
    }
    return(stemList)
  } else {
    return(words)
  }
}

t <- Sys.time()
for (i in c("Internship_Profile")) {
  print(i)
  #internship[[i]] <- cleanUpText(internship[[i]], stem = TRUE)
  internship[[i]] <- sapply(internship[[i]], cleanUpText)
}
print(Sys.time()-t)

#Save file
write.csv(internship, "../data/Internship_Processed.csv", row.names = FALSE)

