setwd("C:/Users/ladwi/Documents/Projects/R/LakePIAB")
library(tidyverse)

data <- read_csv('input/observed_df_lter_hourly_wide.csv')

df <- data[,-c(1,2)]

idx <- which(df < -10 & df > -999, arr.ind = T)

temp_pts <- c()
for (i in 1:nrow(idx)){
  temp_pts <- append(temp_pts, as.numeric(df[idx[i,1], idx[i,2]]))
}

plot(temp_pts)

for (i in 1:nrow(idx)){
  df[idx[i,1], idx[i,2]] = -999
}

data[,3:ncol(data)] = df

data = as.data.frame(data)

data <- write.csv(data, 'input/observed_df_lter_hourly_wide_clean.csv', row.names = F)
