library(dplyr)
library(tidyr)
library(readr)

#LOAD DATA
internship <- read_csv("../data/Internship.csv", na = c("", "NA", "NULL"))
NCOL <- ncol(internship)

#With the below code we checked how the words look like in the Skills column
unlist(strsplit(unlist(strsplit(internship$Skills_required, " ")), ",")) %>% 
  table() %>%
  data.frame() %>% 
  arrange(-Freq) %>% 
  mutate(perc.weight = percent_rank(Freq)) %>%
  filter(perc.weight > 0.95) -> aList

aList$NCHAR <- nchar(as.character(aList$.))
aList <- filter(aList, NCHAR > 1)
StringsForSkills <- setdiff(as.character(aList$.), stopwords("english"))

#Add 4 columns to Student dataframe
internship$Skills_requiredCode <- NA

for (i in StringsForSkills) {
  print(i)
  internship$Skills_requiredCode[grep(i, internship$Skills_required, ignore.case = TRUE)] <- i
}

##Dummy Variables for StreamsCode
for (i in c("Skills_requiredCode")) {
  print(i)
  for(level in unique(internship[[i]])){
    internship[paste("dummy", i, level, sep = "_")] <- ifelse(internship[[i]] == level, 1, 0)
  }
  internship[[i]] <- NULL #Drop this column
}


#SAVE FILES
write.csv(internship[, (NCOL+1):ncol(internship)], "../data/Features_internship_SkillsCode.csv", row.names = F)

