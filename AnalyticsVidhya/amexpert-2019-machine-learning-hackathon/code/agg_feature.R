library(tidyverse)
library(lubridate)

campaign_data <- read_csv("../input/campaign_data.csv")
campaign_data$start_date <- dmy(campaign_data$start_date)
campaign_data$end_date <- dmy(campaign_data$end_date)
campaign_data <- arrange(campaign_data, start_date)


customer_transaction_data <- read_csv("../input/customer_transaction_data.csv")


x <- unique(customer_transaction_data$date)[1]
campaignDates <- campaign_data$start_date
roundToNearestCampaignDate <- function(x) {
  campaignDates[campaignDates > x][1]
}

df_dates <- tibble(date = unique(customer_transaction_data$date))
df_dates <- df_dates %>%
  rowwise() %>%
  mutate(nextCampaignDate = roundToNearestCampaignDate(date))

customer_transaction_data <- left_join(customer_transaction_data, df_dates, by = "date")

customer_transaction_df <- customer_transaction_data %>%
  #head(100000) %>%
  group_by(nextCampaignDate, customer_id, item_id) %>%
  summarise(quantity_sum = sum(quantity, na.rm = T),
            selling_price_sum = sum(selling_price, na.rm = T),
            other_discount_sum = sum(other_discount, na.rm = T),
            coupon_discount_sum = sum(coupon_discount, na.rm = T))

write_csv(customer_transaction_df, "../input/customer_transaction_df.csv")
