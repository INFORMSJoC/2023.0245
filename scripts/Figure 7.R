# Load required libraries
library(ggplot2)
library(plyr)
library(scales)

# Read and preprocess data
df <- read.csv('ADPerf_ExcMyp_m5_LB_Rep4000.csv') 

# Convert 'CS' and 'Endog.' columns to factors with specific levels
df$CS <- factor(df$CS, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))
df$Endog. <- factor(df$Endog., levels = c(
  "[1.6, 2.6, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0]",
  "[1.9, 3.1, 3.1, 2.5, 0.0, 0.0, 0.0, 0.0]",
  "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]",
  "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.08, -0.09]",
  "[1.9, 3.1, 3.1, 2.5, -0.05, -0.1, -0.15, -0.2]",
  "[1.9, 3.1, 3.1, 2.5, -0.1, -0.2, -0.3, -0.4]"
))

# Adjust values for 'Cost', 'Low', and 'High' based on iteration
for (i in 1:nrow(df)) {
  if (df$Iteration[i] == 0) idx <- i
  if (df$Cost[i] > df$Cost[idx]) {
    df$Cost[i] <- df$Cost[idx]
    df$Low[i] <- df$Low[idx]
    df$High[i] <- df$High[idx]
  } else {
    idx <- i
  }
}

# Map 'Endog.' values to more descriptive labels
df$Endog. <- mapvalues(df$Endog., from = c(
  "[1.6, 2.6, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0]",
  "[1.9, 3.1, 3.1, 2.5, 0.0, 0.0, 0.0, 0.0]",
  "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]",
  "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.08, -0.09]",
  "[1.9, 3.1, 3.1, 2.5, -0.05, -0.1, -0.15, -0.2]",
  "[1.9, 3.1, 3.1, 2.5, -0.1, -0.2, -0.3, -0.4]"
), to = c(
  "Exogenous",
  "(1.9, 3.1, 3.1, 2.5)\n(0.0, 0.0, 0.0, 0.0)",
  "Endogenous", "Sensitivity 1",
  "Sensitivity 2", "Sensitivity 3"
))

# Map 'CS' values to more descriptive labels
df$CS <- mapvalues(df$CS, from = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
), to = c(
  "(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)",
  "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"
))

# Filter out unwanted rows
df <- df[df$Endog. != "(1.9, 3.1, 3.1, 2.5)\n(0.0, 0.0, 0.0, 0.0)",]

# Define a function for integer breaks in y-axis
integer_breaks <- function(n = 5, ...) {
  breaker <- pretty_breaks(n, ...)
  function(x) {
    breaks <- breaker(x)
    breaks[breaks == floor(breaks)]
  }
}

# Plot the data
ggplot(df[df$Iteration < 15, ]) +
  geom_line(aes(x = Iteration, y = Cost), color = "black", size = 1) + 
  geom_point(aes(x = Iteration, y = Cost), color = "black", size = 2) + 
  geom_ribbon(aes(x = Iteration, ymin = Low, ymax = High), alpha = 0.2) +
  geom_hline(aes(yintercept = LB), linetype = 'longdash', size = 1, color = "red") +
  geom_line(aes(x = Iteration, y = LB), color = "red", size = 1) +
  geom_point(aes(x = Iteration, y = LB), color = "red", size = 2) +
  geom_ribbon(aes(x = Iteration, ymin = LBLow, ymax = LBHigh), alpha = 0.2, linetype = 0, fill = "red", color = "red") +
  facet_grid(CS ~ Endog., scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 16, 2)) +
  scale_y_continuous(breaks = integer_breaks()) +
  theme_bw(base_size = 30) +
  theme(
    axis.line = element_line(color = 'black'),
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.background = element_rect(size = 0.5, linetype = "solid", color = "black")
  ) +
  labs(y = 'Cost', x = 'Iteration')

# Save the plot as a PDF
ggsave("Figure 7.pdf", width = 20, height = 18, units = "in")
