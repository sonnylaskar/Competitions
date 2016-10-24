#Feature
#Add if InternLocation matches with hometomeLocation, 
#if InternLocation matches with InstitudeLocationCode
#if InternLocation matches with PreferredLocationCode

df_all$isIntern_Loc_Match_HomeTown <- ifelse(df_all$LocationCode == df_all$hometownLocationCode, 1, 0)
df_all$isIntern_Loc_Match_InstitudeLocationCode <- ifelse(df_all$LocationCode == df_all$InstitudeLocationCode, 1, 0)
df_all$isIntern_Loc_Match_PreferredLocationCode <- ifelse(df_all$LocationCode == df_all$PreferredLocationCode, 1, 0)
