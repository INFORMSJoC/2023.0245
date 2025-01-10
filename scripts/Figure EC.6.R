# Load required libraries
library(dplyr)    # For data manipulation
library(ggplot2)  # For data visualization
library(lubridate) # For working with dates
library(nnet)

# # Read the data from a CSV file and convert date columns to proper Date format
# DataPLT <- read.csv('Clean_PLT.csv', header = TRUE, sep = ",", na.strings = c("NA", " "))
# DataPLT$CollectDate = as.Date(DataPLT$CollectDate,  format="%Y-%m-%d")
# DataPLT$ReceiveDate = as.Date(DataPLT$ReceiveDate,  format="%Y-%m-%d")
# DataPLT$ExpiryDate = as.Date(DataPLT$ExpiryDate, format="%Y-%m-%d")
# DataPLT$IssueDT = as.Date(DataPLT$IssueDT, format="%Y-%m-%d")
# DataPLT$StatusChangeDT = as.Date(DataPLT$StatusChangeDT, format="%Y-%m-%d")
# 
# # Filter for data at the "ML" site and from 2015 onwards
# DataPLT_ML=DataPLT %>% filter(year(ReceiveDate) >= 2015 & Site =="ML")
# DataPLT_ML = DataPLT_ML %>% filter(Type == "Original")
# 
# # Calculate shelf-life: difference between ExpiryDate and ReceiveDate (+1 to include last day)
# DataPLT_ML$shelflife <- as.numeric(DataPLT_ML$ExpiryDate - DataPLT_ML$ReceiveDate + 1)
# 
# # Convert ReceiveTime to numeric hour (working with time strings)
# DataPLT_ML$ReceiveTime = as.character(DataPLT_ML$ReceiveTime)
# DataPLT_ML$WHourS = hour(fast_strptime(DataPLT_ML$ReceiveTime, "%H:%M:%OS"))
# 
# # Group by date, hour, product, and unit number; calculate remaining age (RemAge)
# tordata = DataPLT_ML %>% filter(year(ReceiveDate) >= 2017) %>% group_by(ReceiveDate, WHourS, ProductName,ProductUnitNum) %>% summarise(RemAge = shelflife)
# 
# # Calculate daily totals
# trecdata = DataPLT_ML %>% filter(year(ReceiveDate) >= 2017)  %>% group_by(ReceiveDate) %>% summarise(Total = n())
# # Create a complete sequence of dates for 2017
# date <- data.frame(ReceiveDate=seq(as.Date("2017-01-01"), as.Date("2017-12-31"), by="days"))
# # Merge actual data with the complete sequence to fill missing dates with zeros
# trecdata <- merge(trecdata, date, by.x='ReceiveDate', by.y='ReceiveDate', all.x=T, all.y=T)
# trecdata$Total[is.na(trecdata$Total)] = 0
# 
# # Map daily totals (OrderSize) to each RemAge
# tordata$OrderSize <- NA
# for (i in 1:nrow(tordata)){
#   tordata$OrderSize[i] = trecdata$Total[which(trecdata$ReceiveDate == tordata$ReceiveDate[i])]
# }
# 
# # Convert remaining shelf-life values of 6 to 5 and then as a factor
# tordata$RemAge[tordata$RemAge == 6] = 5
# tordata$RemAge = as.factor(tordata$RemAge)
# 
# # Train multinomial logistic regression model
# multi1 <- multinom(RemAge~OrderSize, data = tordata)
# 
# 
# # Subset data for OrderSize = 6
# order6=tordata[which(tordata$OrderSize==6),]
# 
# # Compare historical distribution and model predictions
# df = data.frame(Historical = table(order6$RemAge)/nrow(order6), 
#                 MultiLogitFit = predict(multi1, newdata = data.frame(OrderSize=6), "prob"))
# colnames(df) <- c("ShelfLife", "Historical", "MultiLogitFit")
# write.csv(df, file = 'OrderSize6.csv', row.names = FALSE)
# 
# # Subset data for OrderSize = 8
# order8=tordata[which(tordata$OrderSize==8),]
# 
# # Compare historical distribution and model predictions
# df = data.frame(Historical = table(order8$RemAge)/nrow(order8), 
#                 MultiLogitFit = predict(multi1, newdata = data.frame(OrderSize=8), "prob"))
# colnames(df) <- c("ShelfLife", "Historical", "MultiLogitFit")
# write.csv(df, file = 'OrderSize8.csv', row.names = FALSE)


# Define custom colors for the plot
my_colors <- c("#1f78b4", "#a6cee3")

#----------------------------------------
# Read data for OrderSize = 6 
#----------------------------------------
df <- read.csv('OrderSize6.csv')
# Create a formatted dataframe for plotting
df1 <- data.frame(
  x = c("1","1","2","2", "3", "3", "4","4", "5", "5"),
  y = c(df$Historical[1], df$MultiLogitFit[1], df$Historical[2], df$MultiLogitFit[2],df$Historical[3], df$MultiLogitFit[3],df$Historical[4], df$MultiLogitFit[4],df$Historical[5], df$MultiLogitFit[5]),
  g = rep(c("Data", "Multinomial Logistic Regression"), 5)
)

# Plot comparison for OrderSize = 6
ggplot(data = df1, aes(x, y, group = g, fill = g)) +
  geom_col(position = "dodge", colour = "black") +
  theme_bw(base_size = 30) +
  theme(panel.grid.major = element_blank(), legend.position = "top", legend.title = element_blank()) +
  xlab("Remaining Shelf-life") +
  ylab("Probability") +
  scale_fill_manual(values = my_colors)  # Setting manual colors

# Save the plot as a PDF
ggsave("Figure EC.6 (A).pdf", width = 12, height = 8, units = "in")


#----------------------------------------
# Read data for OrderSize = 8 
#----------------------------------------
df <- read.csv('OrderSize8.csv')
# Create a formatted dataframe for plotting
df1 <- data.frame(
  x = c("1","1","2","2", "3", "3", "4","4", "5", "5"),
  y = c(df$Historical[1], df$MultiLogitFit[1], df$Historical[2], df$MultiLogitFit[2],df$Historical[3], df$MultiLogitFit[3],df$Historical[4], df$MultiLogitFit[4],df$Historical[5], df$MultiLogitFit[5]),
  g = rep(c("Data", "Multinomial Logistic Regression"), 5)
)

# Plot comparison for OrderSize = 8
ggplot(data = df1, aes(x, y, group = g, fill = g)) +
  geom_col(position = "dodge", colour = "black") +
  theme_bw(base_size = 30) +
  theme(panel.grid.major = element_blank(), legend.position = "top", legend.title = element_blank()) +
  xlab("Remaining Shelf-life") +
  ylab("") +
  scale_fill_manual(values = my_colors)  # Setting manual colors

# Save the plot as a PDF
ggsave("Figure EC.6 (B).pdf", width = 12, height = 8, units = "in")
