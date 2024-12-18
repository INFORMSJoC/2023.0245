# Load required libraries
library(lubridate)     # For working with date and time data
library(fitdistrplus)  # For fitting distributions to data

# Read simulated demand data from a CSV file
simulated_data <- read.csv("Simulated_Demand.csv")

# Extract the day of the week from the IssueDT column
simulated_data$day <- wday(simulated_data$IssueDT, label = TRUE)

# Subset the data for Sundays only
simulated_data_Sunday <- simulated_data$Count[which(simulated_data$day == "Fri")]

# Fit a negative binomial distribution to the Sunday data
fnbinom = fitdist(simulated_data_Sunday, "nbinom")

# Fit a Poisson distribution to the Sunday data
fpois = fitdist(simulated_data_Sunday, "pois")

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
