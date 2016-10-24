library(dplyr)
library(tidyr)
library(readr)

#LOAD DATA
student <- read_csv("../data/Student.csv", na = c("", "NA", "NULL"))

#With the below code we checked how the words look like in the degree column
#table(student$Stream) %>% data.frame() %>% arrange(-Freq) %>% View()

NCOL <- ncol(student)
#We will create binary columns for most popular streams

#Add the Temporary Column
student$StreamCode <- NA

StringsForStreams <- c("Computer",
                       "Electronics",
                       "Mechanical",
                       "Commerce",
                       "Information",
                       "Marketing",
                       "Electrical",
                       "Civil",
                       "Finance",
                       "Arts",
                       "Science",
                       "Economics",
                       "Humanities",
                       "Management",
                       "English",
                       "Human",
                       "Software",
                       "Bio",
                       "Mass",
                       "Operations",
                       "Architecture",
                       "Instrumentation",
                       "Mathematics",
                       "Physics",
                       "Media",
                       "Accounts",
                       "Statistics",
                       "Chemistry",
                       "Political Science",
                       "Psychology",
                       "Fashion",
                       "journalism"
                       )

for (i in StringsForStreams) {
  print(i)
  student$StreamCode[grep(i, student$Stream, ignore.case = TRUE)] <- i
}

##Dummy Variables for StreamsCode
for (i in c("StreamCode")) {
  print(i)
  for(level in unique(student[[i]])){
    student[paste("dummy", i, level, sep = "_")] <- ifelse(student[[i]] == level, 1, 0)
  }
  student[[i]] <- NULL #Drop this column
}

#SAVE FILES
write.csv(student[, (NCOL+1):ncol(student)], "../data/Features_student_StreamCode.csv", row.names = F)

