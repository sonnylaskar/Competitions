library(dplyr)
library(tidyr)
library(readr)
library(tm)

#LOAD DATA
student <- read_csv("../data/Student.csv", na = c("", "NA", "NULL"))
NCOL <- ncol(student)

#Filter only the Experience-related columns
student <- student[, c(1,15:19)]

#####SECTION FOR EXPERIENCE ############
#Create columns for each type of Experience and make a single row for each Student ID
student$Experience_Type[is.na(student$Experience_Type)] <- "NoExperience"
student %>%
  select(Student_ID, Experience_Type) %>%
  mutate(yesno = 1) %>%
  distinct %>%
  spread(Experience_Type, yesno, fill = 0) -> studentExperience

#####SECTION FOR PROFILE ############
unlist(strsplit(unlist(strsplit(student$Profile, " ")), ",")) %>% 
  table() %>%
  data.frame() %>% 
  arrange(-Freq) %>% 
  mutate(perc.weight = percent_rank(Freq)) %>%
  filter(perc.weight > 0.98) -> aList

aList$NCHAR <- nchar(as.character(aList$.))
aList <- filter(aList, NCHAR > 1)
aList <- unique(tolower(stemDocument(as.character(aList$.))))
StringsForExperienceProfile <- setdiff(aList, stopwords("english"))

student$Experience_Profile_Type <- NA
for (i in StringsForExperienceProfile) {
  print(i)
  student$Experience_Profile_Type[grep(i, student$Profile, ignore.case = TRUE)] <- i
}

#Create columns for each type of Profile and make a single row for each Student ID
student$Experience_Profile_Type[is.na(student$Experience_Profile_Type)] <- "NoProfile"
student %>%
  select(Student_ID, Experience_Profile_Type) %>%
  mutate(yesno = 1) %>%
  distinct %>%
  spread(Experience_Profile_Type, yesno, fill = 0) -> studentExperienceProfile

#JOIN
studentExperience <- left_join(studentExperience, studentExperienceProfile, by = "Student_ID")
#SAVE FILES
write.csv(studentExperience, "../data/Features_student_Experience.csv", row.names = F)

