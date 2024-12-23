# Load required libraries
library(lubridate)     # For working with date and time data
library(fitdistrplus)  # For fitting distributions to data

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

# Extract the day of the week from the Date column
result$day <- wday(result$Date, label = TRUE)

# Subset the data for Mondays only
result_Monday <- result$Demand[which(result$day == "Mon")]

# Fit a negative binomial distribution to the Monday data
fnbinom = fitdist(result_Monday, "nbinom")

# Fit a Poisson distribution to the Monday data
fpois = fitdist(result_Monday, "pois")

# Set up a 1x2 grid for two plots
par(mfrow = c(1, 2))

# Define legend labels for the plots
plot.legend <- c("nbinom", "poisson")

# Density comparison plot: Compare fitted densities to the data
denscomp(
  list(fnbinom, fpois), 
  legendtext = plot.legend,
  main = "",             
  ylab = "Probability",
  xlab = "Data"
  )


# QQ plot: Compare theoretical quantiles of fitted distributions to observed data
qqcomp(
  list(fnbinom, fpois), 
  legendtext = plot.legend,
  main = ""          
)
