# Load required libraries
library(lubridate) # For working with dates
library(dplyr)    # For data manipulation
library(ggplot2)  # For data visualization

#############################################
########### Simulate Data ###################
#############################################

# Define size and prob parameters for each day of the week
size <- c(3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797)
prob <- c(
  size[1] / (size[1] + 5.660569),
  size[2] / (size[2] + 6.922555),
  size[3] / (size[3] + 6.504332),
  size[4] / (size[4] + 6.165049),
  size[5] / (size[5] + 5.816060),
  size[6] / (size[6] + 3.326408),
  size[7] / (size[7] + 3.426814)
)

# Generate the sequence of dates
dates <- seq(as.Date("2015-01-01"), as.Date("2016-12-31"), by = "day")

# Map days of the week to indices (Monday = 1, Sunday = 7)
day_indices <- wday(dates, week_start = 1)

# Generate random demand for each day
set.seed(123) # For reproducibility
demand <- sapply(day_indices, function(day) {
  rnbinom(1, size = size[day], prob = prob[day])
})

# Combine dates and demands into a data frame
result <- data.frame(Date = dates, Demand = demand)

#############################################

# Add a column for day of the week and reorder levels for proper plotting order
result$day <- wday(result$Date, label = TRUE)
result <- within(result, day <- factor(day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))

# Calculate the overall mean of daily demand
dMean <- result %>%
  summarise(MN = mean(Demand))

# Calculate the mean demand for each day of the week
DOWMean <- result %>%
  group_by(day) %>%
  summarise(MN = mean(Demand), .groups = 'drop')

# Define custom plot colors and transparency
color <- c("Black", "Black")
alpha_color <- c(1, 1)
alpha_fill <- c(0.6, 0.2)

# Save the plot as a PDF with a custom size of 14 x 7.5 inches
result %>%
  ggplot(mapping = aes(x = day, y = Demand)) +
  # Boxplot for daily Demands
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
