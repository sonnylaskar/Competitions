library(readr)

#Ensemble the 2 XGB models
MODEL_1 <- read_csv("../Submissions/XGB_MODEL_S123_N526.csv")
MODEL_2 <- read_csv("../Submissions/XGB_MODEL_S500_N710.csv")

MEANSCORE <- (MODEL_1$Is_Shortlisted + MODEL_2$Is_Shortlisted) / 2

#SAVE
submission <- data.frame(Internship_ID = MODEL_1$Internship_ID,
                         Student_ID = MODEL_1$Student_ID,
                         Is_Shortlisted = MEANSCORE)
write_csv(submission,"../Submissions/FINAL_SUBMISSION.csv")  
