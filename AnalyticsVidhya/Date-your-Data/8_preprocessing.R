library(dplyr)
library(tidyr)
library(readr)

#LOAD DATA
internship <- read_csv("../data/Internship_Processed.csv", na = c("", "NA", "NULL"))
student <- read_csv("../data/Student.csv", na = c("", "NA", "NULL"))
train <- read_csv("../data/train.csv", na = c("", "NA", "NULL"))
test <- read_csv("../data/test.csv", na = c("", "NA", "NULL"))
#Add Is_Shortlisted column to Test
test$Is_Shortlisted <- -1
#combine data
df_all <- rbind(train, test)
NROWS_TRAIN <- nrow(train)


#Built Numeric Codes for all LOCATION CODES
l <- unique(internship$Internship_Location)
l <- append(l, unique(student$Institute_location))
l <- append(l, unique(student$hometown))
l <- append(l, unique(student$Location))
l <- append(l, unique(df_all$Preferred_location))
l <- unique(l)

LocatioNumericCodes <- data.frame(Location = l, Code = 1:length(l))

##############################
# INTERNSHIP DATA

#Define dates as Date Class objects
internship$Start_Date <- as.Date(internship$Start_Date, "%d-%m-%Y")
internship$Internship_deadline <- as.Date(internship$Internship_deadline, "%d-%m-%Y")

#Few values in internship$"Internship_Duration(Months) are unrealistic
#Replacing with 12 months
internship$"Internship_Duration(Months)"[internship$"Internship_Duration(Months)" > 1000] <- 12

#Add Internship End Date
internship$End_Date <- (internship$Start_Date + internship$"Internship_Duration(Months)"*30)

#Split Date columns into Separate Date, Month, Year
internship <- separate(internship, 
                       Internship_deadline, 
                       c("Internship_Deadline_Date_Y", "Internship_Deadline_Date_M", "Internship_Deadline_Date_D"), 
                       sep = "-",
                       remove = FALSE) 
internship <- separate(internship, 
                       Start_Date, 
                       c("Internship_Start_Date_Y", "Internship_Start_Date_M", "Internship_Start_Date_D"), 
                       sep = "-",
                       remove = FALSE) 
internship <- separate(internship, 
                       End_Date, 
                       c("Internship_End_Date_Y", "Internship_End_Date_M", "Internship_End_Date_D"), 
                       sep = "-",
                       remove = FALSE) 

#Add a column for Location Numeric Code
left_join(internship[, "Internship_Location"], LocatioNumericCodes, by = c("Internship_Location"="Location")) %>% 
  mutate(LocationCode = Code) %>%
  select(LocationCode) %>%
  collect %>% .[["LocationCode"]] -> internship$LocationCode
#Remove Original Column
internship$Internship_Location <- NULL

##Dummy Variables
for (i in c("Internship_Type", 
            "Internship_category", 
            "Stipend_Type")) {
  print(i)
  for(level in unique(internship[[i]])){
    internship[paste("dummy", i, level, sep = "_")] <- ifelse(internship[[i]] == level, 1, 0)
  }
  internship[[i]] <- NULL #Drop this column
}

#FEATURES SECTION
#Add Ngram features for Profile
Internship_nGramProfile <- read_csv("../data/Features_internship_Profile_WordCount.csv")
internship <- cbind(internship, Internship_nGramProfile)

Internship_ProfileCode <- read_csv("../data/Features_internship_ProfileCode.csv")
internship <- cbind(internship, Internship_ProfileCode)

Internship_SkillsCode <- read_csv("../data/Features_internship_SkillsCode.csv")
internship <- cbind(internship, Internship_SkillsCode)
#DROP THESE COLUMNS
internship$Internship_Profile <- NULL #ALREADY PROCESSED
internship$Skills_required <- NULL #ALREADY PROCESSED

##############################
# STUDENT DATA
#Define dates as Date Class objects
student$"Start Date" <- as.Date(student$"Start Date", "%d-%m-%Y")
student$"End Date" <- as.Date(student$"End Date", "%d-%m-%Y")

#Add Total Months of experience column
student$MonthsOfExperience <- round(as.numeric((student$"End Date" - student$"Start Date") / 30))
#Split Date columns into Separate Date, Month, Year
student <- separate(student, 
                    "Start Date", 
                    c("Start_Date_Y", "Start_Date_M", "Start_Date_D"), 
                    sep = "-",
                    remove = FALSE) 
student <- separate(student, 
                    "End Date", 
                    c("End_Date_Y", "End_Date_M", "End_Date_D"), 
                    sep = "-",
                    remove = FALSE) 

#Define Performance UG/PG in Percentage since some have CGPA and some have %ages
student$Performance_PG_percent <- student$Performance_PG / student$PG_scale
student$Performance_UG_percent <- student$Performance_UG / student$UG_Scale
#Convert 10/12 Score to Percentages
student$Performance_10th <- student$Performance_10th / 100
student$Performance_12th <- student$Performance_12th / 100
#Also add mean score for 10/12/UG/PG #mean function was giving some error
student$Performance_MEAN <- (student$Performance_PG_percent +
                                 student$Performance_UG_percent + 
                                 student$Performance_10th +
                                 student$Performance_12th) / 4
                                 

#Add a column for Location Numeric Code
left_join(student[, "Institute_location"], LocatioNumericCodes, by = c("Institute_location"="Location")) %>% 
  mutate(LocationCode = Code) %>%
  select(LocationCode) %>%
  collect %>% .[["LocationCode"]] -> student$InstitudeLocationCode
left_join(student[, "hometown"], LocatioNumericCodes, by = c("hometown"="Location")) %>% 
  mutate(LocationCode = Code) %>%
  select(LocationCode) %>%
  collect %>% .[["LocationCode"]] -> student$hometownLocationCode
#Remove Original Column
student$Institute_location <- NULL
student$hometown <- NULL


#DROP THESE FOR LATER
student$Experience_Type <- NULL
student$Profile <- NULL
student$Location <- NULL
student$Degree <- NULL #ALREADY PROCESSED
student$Stream <- NULL #ALREADY PROCESSED

##Dummy Variables
for (i in c("Year_of_graduation", 
            "Current_year", 
            "Institute_Category")) {
  print(i)
  for(level in unique(student[[i]])){
    student[paste("dummy", i, level, sep = "_")] <- ifelse(student[[i]] == level, 1, 0)
  }
  student[[i]] <- NULL #Drop this column
}

#FEATURES SECTION
#Add Ngram features for Profile
student_StreamCode <- read_csv("../data/Features_student_StreamCode.csv")
student <- cbind(student, student_StreamCode)
student_DegreeCode <- read_csv("../data/Features_student_DegreeCode.csv")
student <- cbind(student, student_DegreeCode)

#Remove duplicate rows for Students as of now
student <- student[!duplicated(student[1]),]
#Add Experience Features
student_Experience <- read_csv("../data/Features_student_Experience.csv")
student <- left_join(student, student_Experience, by = "Student_ID")
##############################
# TRAIN /  TEST DATA


#Define dates as Date Class objects
df_all$Earliest_Start_Date <- as.Date(df_all$Earliest_Start_Date, "%d-%m-%Y")

#Define Ordered Factor variables for Expectated Stipend
df_all$Expected_Stipend <- factor(df_all$Expected_Stipend, 
                                  c("No Expectations", "2-5K", "5-10K", "10K+"), 
                                  order = T)
#Add a column for Location Numeric Code
left_join(df_all[, "Preferred_location"], LocatioNumericCodes, by = c("Preferred_location"="Location")) %>% 
  mutate(LocationCode = Code) %>%
  select(LocationCode) %>%
  collect %>% .[["LocationCode"]] -> df_all$PreferredLocationCode
#Remove Original Column
df_all$Preferred_location <- NULL

#Split Date columns into Separate Date, Month, Year
df_all <- separate(df_all, 
                   Earliest_Start_Date, 
                   c("Earliest_Start_Date_Y", "Earliest_Start_Date_M", "Earliest_Start_Date_D"), 
                   sep = "-",
                   remove = FALSE) 

##Dummy Variables
for (i in c("Expected_Stipend")) {
  print(i)
  for(level in unique(df_all[[i]])){
    df_all[paste("dummy", i, level, sep = "_")] <- ifelse(df_all[[i]] == level, 1, 0)
  }
  df_all[[i]] <- NULL #Drop this column
}

####### ADD INTERNSHIP/STUDENT DATA TO DF_ALL
df_all <- left_join(df_all, internship, by = "Internship_ID")
df_all <- left_join(df_all, student, by = "Student_ID")

##########################
## FEATURES
#Add features regarding whether Student applied before Start_Date, End_Date and Deadline
df_all$Earliest_Date_Before_Start_Date <- as.integer(df_all$Earliest_Start_Date < df_all$Start_Date)
df_all$Earliest_Date_Before_End_Date <- as.integer(df_all$Earliest_Start_Date < df_all$End_Date)
df_all$Earliest_Date_Before_Deadline_Date <- as.integer(df_all$Earliest_Start_Date < df_all$Internship_deadline)

#Add count of Internships Applications
source("../Code/feature_df_all_CountOfApplications.R")

#Add if InternLocation matches with hometomeLocation, 
#if InternLocation matches with InstitudeLocationCode
#if InternLocation matches with PreferredLocationCode
source("../Code/feature_df_all_Match_Internship_Location_with_other_locations.R")
###########################
#Bring target Column at the front
df_all <- df_all[, c("Is_Shortlisted", setdiff(names(df_all), "Is_Shortlisted"))]

#Split Train/validate/Test Set
train <- df_all[1:NROWS_TRAIN, ]
test <- df_all[-(1:NROWS_TRAIN), ]

#Save files
write.csv(train, "../data/train_processed.csv", row.names = FALSE)
write.csv(test, "../data/test_processed.csv", row.names = FALSE)