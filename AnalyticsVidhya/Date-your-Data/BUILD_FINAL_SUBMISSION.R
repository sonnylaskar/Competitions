#This will build the Final Solution
#Will take some time

source("1_internship_WordCorrection.R")
source("2_feature_internship_Profile_WordCount.R")
source("3_feature_internship_Profile_Coding.R")
source("4_feature_internship_SkillsCoding.R")
source("5_feature_student_StreamsCoding.R")
source("6_feature_student_degreeCoding.R")
source("7_feature_student_ExperienceCoding.R")
source("8_preprocessing.R")

print("Building First XGB model")
source("9_model_XGB_1.R")
print("Building Second XGB model")
source("10_model_XGB_1.R")

print("Calculating the Average of the 2 models")
source("11_Ensemble_Models.R")
print("Huh!!!, I am done!")
print("Check out FINAL_SUBMISSION FILE in Submission FOlder")