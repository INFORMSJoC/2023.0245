# Load required libraries
library(dplyr)    # For data manipulation
library(ggplot2)  # For data visualization
library(lubridate) # For working with dates

# # Load the real dataset and parse date columns
# DataPLT <- read.csv('Clean_PLT.csv', header = TRUE, sep = ",", na.strings = c("NA", " "))
# DataPLT$CollectDate <- as.Date(DataPLT$CollectDate, format = "%Y-%m-%d")
# DataPLT$ReceiveDate <- as.Date(DataPLT$ReceiveDate, format = "%Y-%m-%d")
# DataPLT$ExpiryDate <- as.Date(DataPLT$ExpiryDate, format = "%Y-%m-%d")
# DataPLT$StatusChangeDT <- as.Date(DataPLT$StatusChangeDT, format = "%Y-%m-%d")
# DataPLT$IssueDT <- as.Date(DataPLT$IssueDT, format = "%Y-%m-%d")
# 
# # Filter data for the site 'ML' (Hamilton General Hospital) and years 2015 and 2016
# DataPLT_ML <- DataPLT %>%
#   filter(year(IssueDT) %in% c(2015, 2016), Site == "ML")
# 
# # Summarize daily counts
# tdemdata <- DataPLT_ML %>%
#   filter(year(IssueDT) >= 2015 & !is.na(IssueDT)) %>%
#   group_by(IssueDT) %>%
#   summarise(Count = n(), .groups = 'drop')
# 
# # Create a complete date range for 2015-2016 and merge with summarized data
# date <- data.frame(IssueDT = seq(as.Date("2015-01-01"), as.Date("2016-12-31"), by = "days"))
# tdemdata <- merge(tdemdata, date, by = 'IssueDT', all.x = TRUE, all.y = TRUE)
# 
# # Fill missing values with defaults
# tdemdata$Count[is.na(tdemdata$Count)] <- 0
# 
# # Add a column for day of the week and reorder levels for proper plotting order
# tdemdata$day <- wday(tdemdata$IssueDT, label = TRUE)
# tdemdata <- within(tdemdata, day <- factor(day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))
# 
# #############################################
# ########### Simulate Data ###################
# #############################################
# set.seed(123)  # For reproducibility
# 
# # Simulate new counts by sampling within the same day of the week
# simulated_data <- data.frame(
#   IssueDT = tdemdata$IssueDT,  # Keep the original dates
#   Count = sapply(tdemdata$day, function(d) {
#     # Randomly sample a count from the same day of the week
#     sample(tdemdata$Count[tdemdata$day == d], 1)
#   })
# )
# 
# write.csv(simulated_data, file = "Simulated_Demand.csv", row.names = FALSE)
# #############################################

# Read simulated demand data
tdemdata <- read.csv("Simulated_Demand.csv")
# Add a column for day of the week and reorder levels for proper plotting order
tdemdata$day <- wday(tdemdata$IssueDT, label = TRUE)
tdemdata <- within(tdemdata, day <- factor(day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))

# Calculate the overall mean of daily demand
dMean <- tdemdata %>%
  summarise(MN = mean(Count))

# Calculate the mean demand for each day of the week
DOWMean <- tdemdata %>%
  group_by(day) %>%
  summarise(MN = mean(Count), .groups = 'drop')

# Define custom plot colors and transparency
color <- c("Black", "Black")
alpha_color <- c(1, 1)
alpha_fill <- c(0.6, 0.2)

# Save the plot as a PDF with a custom size of 14 x 7.5 inches
tdemdata %>%
  ggplot(mapping = aes(x = day, y = Count)) +
  # Boxplot for daily counts
  geom_boxplot() +
  # Add mean points for each day of the week
  geom_point(data = DOWMean,
             mapping = aes(x = day, y = MN),
             color = "blue", size = 3) +
  # Line connecting the mean points
  geom_line(data = DOWMean,
            mapping = aes(x = day, y = MN, group = 1),
            color = "blue", size = 0.8) +
  # Horizontal line for overall mean
  geom_hline(data = dMean, aes(yintercept = MN),
             linetype = 'longdash', colour = "red", size = 0.8) +
  # Clean white background and customize font size
  theme_bw(base_size = 35) +
  theme(axis.title.x = element_blank()) +
  # Customize y-axis scale
  scale_y_continuous(breaks = seq(0, 30, 2)) +
  ylab("Daily demand") +
  # Remove legend
  theme(legend.position = 'none') +
  # Apply custom color scales with transparency
  scale_fill_manual(values = alpha(color, alpha_fill)) +
  scale_color_manual(values = alpha(color, alpha_color))

# Save the plot as a PDF
ggsave("Figure 11 - Simulated.pdf", width = 14, height = 7.5, units = "in")
