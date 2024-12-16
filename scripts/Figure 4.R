# Load required libraries
library(ggplot2)
library(plyr)
library(scales)

# ####################### Data Preprocessing #######################

# Load the initial dataset
df <- read.csv('ADPerformanceOptGap_Myopic_Exact.csv')

# Define and order levels for Cost Structures (CS)
df$CS <- factor(df$CS, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))

# Define and order levels for Endogeneity (Endog.)
df$Endog. <- factor(df$Endog., levels = c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
  "[1, 0.5, -0.2, -0.4]", "[1, 0.5, -0.4, -0.8]"
))

# Filter relevant rows for initial results
df1 <- df[df$Endog. %in% c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]",
  "[1, 0.5, 0.0, 0.0]", "[1, 0.5, -0.4, -0.8]"
), ]

# Load the extended dataset
df <- read.csv('ADPerformanceOptGap_Myopic_Exact_NegParams.csv')

# Define and order levels for Cost Structures (CS) in the extended dataset
df$CS <- factor(df$CS, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))

# Define and order levels for Endogeneity (Endog.) in the extended dataset
df$Endog. <- factor(df$Endog., levels = c(
  "[1, 0.5, 0.0, 0.0]", "[1, 0.5, -0.05, -0.1]",
  "[1, 0.5, -0.1, -0.2]", "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]"
))

# Filter relevant rows for extended results
df2 <- df[df$Endog. %in% c(
  "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]"
), ]

# Combine filtered dataframes
df <- rbind(df2, df1)

# Re-define factor levels for consistency in the combined dataset
df$CS <- factor(df$CS, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))

df$Endog. <- factor(df$Endog., levels = c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
  "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"
))

# ####################### Update Optimality Gap #######################

# Track the best performance so far across iterations
for (i in 1:nrow(df)) {
  if (df$Iteration[i] == 0) idx <- i
  if (df$OptGap[i] > df$OptGap[idx]) {
    # Update values to the best observed so far
    df$OptGap[i] <- df$OptGap[idx]
    df$Cost[i] <- df$Cost[idx]
    df$Low[i] <- df$Low[idx]
    df$High[i] <- df$High[idx]
  } else {
    idx <- i
  }
}

# ####################### Prepare Data for Plotting #######################

# Map Endogeneity and Cost Structures to user-friendly names
df$Endog. <- mapvalues(df$Endog., from = c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
  "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"
), to = c(
  "(1, 0.5, 0.4, 0.8)", "(1, 0.5, 0.2, 0.4)", "(1, 0.5, 0.0, 0.0)",
  "(1, 0.5, -0.1, -0.05)", "(1, 0.5, -0.2, -0.1)", "(1, 0.5, -0.4, -0.8)"
))

df$CS <- mapvalues(df$CS, from = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
), to = c(
  "(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)",
  "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"
))

# Custom integer breaks for y-axis
integer_breaks <- function(n = 5, ...) {
  breaker <- pretty_breaks(n, ...)
  function(x) {
    breaks <- breaker(x)
    breaks[breaks == floor(breaks)]
  }
}

# Add a scaling variable for y-axis adjustments
df$scale <- c(rep(5, 2 * 21), rep(10, 2 * 21), rep(72, 2 * 21),
              rep(23, 2 * 21), rep(15, 2 * 21), rep(10, 2 * 21),
              rep(5, 4 * 21), rep(10, 4 * 21), rep(72, 4 * 21),
              rep(23, 4 * 21), rep(15, 4 * 21), rep(10, 4 * 21))

# ####################### Create and Save Plot #######################

# Plot Optimality Gap over iterations with facets for CS and Endog.
ggplot(df[df$Iteration < 15, ], aes(x = Iteration, y = OptGap)) +
  geom_line(color = "black", size = 1) +
  geom_point(color = "black", size = 2) +
  geom_ribbon(aes(ymin = Low, ymax = High), alpha = 0.2) +
  geom_point(aes(x = Iteration, y = scale), color = "white") +  # Trick for free y-scaling
  facet_grid(CS ~ Endog., scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 12, 4)) +
  scale_y_continuous(breaks = integer_breaks()) +
  theme_bw(base_size = 30) +
  theme(
    axis.line = element_line(color = "black"),
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")
  ) +
  labs(y = "Optimality Gap (%)", x = "Iteration")

# Save plot as PDF with custom size
ggsave("Figure 4.pdf", width = 22, height = 17)
