# Load required libraries
library(dplyr)    # For data manipulation
library(ggplot2)  # For data visualization
library(lubridate) # For working with dates

# Initialize an empty dataframe to accumulate results
df1 <- data.frame()

# Function to process datasets
process_dataset <- function(file_path, fixed_cost, expiration_cost, policy) {
  # Load data from file
  df <- read.csv(file_path)
  
  # Assign dates and calculate weekdays
  df$Date <- seq(as.Date("2017-01-01"), as.Date("2017-12-31"), by = "days")
  df$day <- wday(df$Date, label = TRUE)
  
  # Summarize and calculate mean values per day
  df_summary <- df %>%
    group_by(day) %>%
    summarize(across(where(is.numeric), mean)) %>%
    mutate(mean = rowMeans(pick(where(is.numeric)))) %>%
    as.data.frame()
  
  # Keep necessary columns and add metadata
  df_summary <- df_summary %>%
    select(day, mean) %>%
    mutate(f = paste("Fixed ordering cost =", fixed_cost),
           theta = paste("Unit expiration cost =", expiration_cost),
           policy = policy)
  
  return(df_summary)
}

# Process datasets with different parameters
# Parameters: file_path, fixed_cost, expiration_cost, policy
datasets <- list(
  list("Deter102.csv", 10, 2, "Deter."),
  list("Exog102.csv", 10, 2, "Exoge."),
  list("Endog102.csv", 10, 2, "Endog."),
  list("Deter105.csv", 10, 5, "Deter."),
  list("Exog105.csv", 10, 5, "Exoge."),
  list("Endog105.csv", 10, 5, "Endog."),
  list("Deter1020.csv", 10, 20, "Deter."),
  list("Exog1020.csv", 10, 20, "Exoge."),
  list("Endog1020.csv", 10, 20, "Endog."),
  list("Deter202.csv", 20, 2, "Deter."),
  list("Exog202.csv", 20, 2, "Exoge."),
  list("Endog202.csv", 20, 2, "Endog."),
  list("Deter205.csv", 20, 5, "Deter."),
  list("Exog205.csv", 20, 5, "Exoge."),
  list("Endog205.csv", 20, 5, "Endog."),
  list("Deter2020.csv", 20, 20, "Deter."),
  list("Exog2020.csv", 20, 20, "Exoge."),
  list("Endog2020.csv", 20, 20, "Endog.")
)

# Apply processing function to each dataset
for (dataset in datasets) {
  df1 <- rbind(df1, process_dataset(dataset[[1]], dataset[[2]], dataset[[3]], dataset[[4]]))
}

# # Process PLT data
# load("~/PLT_ML_Prediction/PLTdata_ML_v3.rda")
# plt_data <- tdemdata[732:1096,]
# plt_data$Date <- seq(as.Date("2017-01-01"), as.Date("2017-12-31"), by = "days")
# plt_data = plt_data[c(1,54)]
# plt_data$day <- wday(plt_data$Date, label = TRUE)
# 
# # Calculate mean values and append PLT data for each scenario
# plt_summary <- plt_data %>%
#   group_by(day) %>%
#   summarize(across(.cols = where(is.numeric),.fns = mean)) %>%
#   mutate(mean = rowMeans(pick(where(is.numeric)))) %>% as.data.frame()
# 
# plt_summary <- plt_summary[c("day", "mean")]
# write.csv(plt_summary, 'Average_Demand.csv', row.names = FALSE)

plt_summary = read.csv('Average_Demand.csv')

fixed_costs <- c(10, 20)
expiration_costs <- c(2, 5, 20)

for (f in fixed_costs) {
  for (theta in expiration_costs) {
    temp <- plt_summary %>%
      mutate(f = paste("Fixed ordering cost =", f),
             theta = paste("Unit expiration cost =", theta),
             policy = "Demand")
    df1 <- rbind(df1, temp)
  }
}

df1$day = factor(df1$day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
df1$policy = factor(df1$policy, levels = c("Demand", "Endog.", "Exoge.", "Deter."))
df1$theta = factor(df1$theta, levels = c("Unit expiration cost = 2", "Unit expiration cost = 5", "Unit expiration cost = 20"))
df1$f = factor(df1$f, levels = c("Fixed ordering cost = 20", "Fixed ordering cost = 10"))

# Plot the results
ggplot(df1, aes(x = day, y = mean, color = policy, group = policy)) + 
  geom_line(size = 1) + 
  geom_point(size = 3) + 
  facet_grid(f ~ theta) + 
  theme_bw(base_size = 30) + 
  labs(
    x = "", 
    y = "Average Target Inventory Levels", 
    color = NULL
  ) +
  theme(
    axis.text.x = element_text(angle = 0), 
    legend.position = "top", 
    legend.title = element_blank()
  ) +
  scale_y_continuous(breaks = seq(0, 18, 2), limits = c(2, 18)) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE))

# Save the plot as a PDF
ggsave("Figure 12.pdf", width = 20, height = 14, units = "in")
