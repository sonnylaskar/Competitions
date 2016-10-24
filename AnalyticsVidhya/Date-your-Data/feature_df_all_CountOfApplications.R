#Feature
#Add a column of how many applications received for any Internship_ID
Intern_Freq <- data.frame(table(df_all$Internship_ID))
names(Intern_Freq) <- c("Internship_ID", "Internship_ApplicationCount")
Intern_Freq$Internship_ID <- as.integer(as.character(Intern_Freq$Internship_ID))
df_all <- left_join(df_all, Intern_Freq, by = "Internship_ID" )
rm(Intern_Freq)
#END