library(tidyverse)
library(lubridate)

createDf <- function(file_names) {
  df <- tibble()
  for (i in file_names) {
    tmp <- suppressMessages(read_csv(i))
    if (nrow(df) == 0) {
      df <- tmp
    } else {
      df <- left_join(df, tmp, by = c("CampaignDate", "customer_id", "item_id"))
    }
    rm(tmp)
    gc()
  }
  df
}

#[1] 26 27 28 29 30  1  2  3  4  5  6  7  8  9 10 11 12 13 16 17 18 19 20 21 22 23 24 25
#assign("df_2", df_1)

df <- tibble()
for (i in 1:30) {
  print(i)
  tmp <- createDf(list.files(path = "../input/",
                              pattern = paste0("agg_feat_",i,"_"),
                              full.names = T))
  #assign(paste0("df_", i), df)
  if (nrow(df) == 0) {
    df <- tmp
  } else {
    df <- bind_rows(df, tmp)
  }
  rm(tmp)
  gc()
}

write_csv(df, "../input/agg_v2.csv")

