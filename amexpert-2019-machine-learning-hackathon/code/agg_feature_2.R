library(tidyverse)
library(lubridate)

campaign_data <- read_csv("../input/campaign_data.csv")
campaign_data$start_date <- dmy(campaign_data$start_date)
campaign_data$end_date <- dmy(campaign_data$end_date)
campaign_data <- arrange(campaign_data, start_date)


customer_transaction_data <- read_csv("../input/customer_transaction_data.csv")


#x <- unique(customer_transaction_data$date)
#campaignDates <- campaign_data$start_date
#roundToNearestCampaignDate <- function(x) {
#  campaignDates[campaignDates > x][1]
#}

#df_dates <- tibble(date = unique(customer_transaction_data$date))
#df_dates <- df_dates %>%
#  rowwise() %>%
#  mutate(nextCampaignDate = roundToNearestCampaignDate(date))

#customer_transaction_data <- left_join(customer_transaction_data, df_dates, by = "date")

#customer_transaction_df <- customer_transaction_data %>%
  #head(100000) %>%
#  group_by(nextCampaignDate, customer_id, item_id) %>%
#  summarise(quantity_sum = sum(quantity, na.rm = T),
#            selling_price_sum = sum(selling_price, na.rm = T),
#            other_discount_sum = sum(other_discount, na.rm = T),
#            coupon_discount_sum = sum(coupon_discount, na.rm = T))

#write_csv(customer_transaction_df, "../input/customer_transaction_df.csv")

#df_dates <- tibble(campaignDates = campaignDates)
#df_dates$date_1m <- df_dates$campaignDates - 30
#df_dates$date_2m <- df_dates$campaignDates - 60

for (i in unique(campaign_data$campaign_id)) {
  customer_transaction_data[[paste0("campaign_id_", i)]] <- campaign_data$start_date[campaign_data$campaign_id == i]
}

#[1] 26 27 28 29 30  1  2  3  4  5  6  7  8  9 10 11 12 13 16 17 18 19 20 21 22 23 24 25

#customer_transaction_df <- tibble()
for (i in unique(campaign_data$campaign_id)) {
  for (lagDays in c(seq(30, 30*12, 30))) {
    print(paste(i, lagDays))
    customer_transaction_data$CampaignDate <- customer_transaction_data[[paste0("campaign_id_", i)]]
    tmp <- customer_transaction_data %>%
      filter(date >= CampaignDate - lagDays & date < CampaignDate) %>%
      group_by(CampaignDate, customer_id, item_id) %>%
      summarise(quantity_sum = sum(quantity, na.rm = T),
                selling_price_sum = sum(selling_price, na.rm = T),
                other_discount_sum = sum(other_discount, na.rm = T),
                coupon_discount_sum = sum(coupon_discount, na.rm = T),
                quantity_mean = mean(quantity, na.rm = T),
                selling_price_mean = mean(selling_price, na.rm = T),
                other_discount_mean = mean(other_discount, na.rm = T),
                coupon_discount_mean = mean(coupon_discount, na.rm = T))
    
    
    if (nrow(tmp) > 0) {
      names(tmp)[-(1:3)] <- paste(names(tmp)[-(1:3)], lagDays, sep = "_")
      #customer_transaction_df <- bind_rows(customer_transaction_df, tmp)
      write_csv(tmp, paste("../input/agg_feat",i, lagDays, ".csv", sep = "_"))
      rm(tmp)
    }
    gc()
  }  
}

#write_csv(customer_transaction_df, "../input/agg_feat_2.csv")
