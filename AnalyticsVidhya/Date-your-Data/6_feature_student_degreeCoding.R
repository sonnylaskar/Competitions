library(dplyr)
library(tidyr)
library(readr)

#LOAD DATA
student <- read_csv("../data/Student.csv", na = c("", "NA", "NULL"))

#With the below code we checked how the words look like in the degree column
#table(student$Degree) %>% data.frame() %>% arrange(-Freq) %>% View()

#We will create 4 binary columns to identify  the following:
#1) IsUnderGraduate
#2) IsPostGraduate
#3) IsTechbackground
#4) IsNonTechbackground

StringsForUG <- c("BE|B.|Bachelor|Undergrad|BCA|UG|BBA|LLB")

StringsForPG <- c("MBA|Management|M.|MCA|MBA|Post Graduate|Master|Ph.D")

StringsForTech <- c("MCA|M.Tech|M. Tech|BCA|B.E.|B. E.|B.Tech|B. Tech|Science|Technology|Engineer|Software")

StringsForNonTech <- c("MBA|Management|BBA|LLB|Business|Journalism|Mass|Arts|Pharma|Chartered|Dental|Social|English|Finance|Sports|Media|Fashion|Psychology")

NCOL <- ncol(student)
#Add 4 columns to Student dataframe
student$IsUnderGraduate <- 0
student$IsPostGraduate <- 0
student$IsTechbackground <- 0
student$IsNonTechbackground <- 0

student$IsUnderGraduate[grep(StringsForUG, student$Degree, ignore.case = TRUE)] <- 1
student$IsPostGraduate[grep(StringsForPG, student$Degree, ignore.case = TRUE)] <- 1
student$IsTechbackground[grep(StringsForTech, student$Degree, ignore.case = TRUE)] <- 1
student$IsNonTechbackground[grep(StringsForNonTech, student$Degree, ignore.case = TRUE)] <- 1

#SAVE FILES
write.csv(student[, (NCOL+1):ncol(student)], "../data/Features_student_DegreeCode.csv", row.names = F)

