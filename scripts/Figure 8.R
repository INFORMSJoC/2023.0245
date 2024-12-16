# Load necessary libraries
library(ggplot2)
library(plyr)
library(scales)

# Load data
df <- read.csv('ADPerf_m8_RelGap.csv')

# Define factor levels for 'CS' and 'Endog.' columns
cs_levels <- c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", 
               "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]")
cs_labels <- c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", 
               "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)")

endog_levels <- c("[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
                  "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09]",
                  "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8, -0.06, -0.08, -0.1, -0.12, -0.14, -0.16, -0.18]",
                  "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8, -0.12, -0.16, -0.2, -0.24, -0.28, -0.32, -0.36]",
                  "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8, -0.24, -0.32, -0.4, -0.48, -0.56, -0.64, -0.72]",
                  "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8, -0.48, -0.64, -0.8, -0.96, -1.12, -1.28, -1.44]")
endog_labels <- c("Exogenous", "Endogenous", "Sensitivity 1", "Sensitivity 2", "Sensitivity 3", 
                  "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8]\n[-0.48, -0.64, -0.8, -0.96, -1.12, -1.28, -1.44]")

# Convert to factors with meaningful labels
df$CS <- factor(df$CS, levels = cs_levels, labels = cs_labels)
df$Endog. <- factor(df$Endog., levels = endog_levels, labels = endog_labels)

# Update the iteration and gap values
idx <- 1
for (i in seq_len(nrow(df))) {
  if (df$Iteration[i] == 0) idx <- i
  if (df$EGap[i] < df$EGap[idx]) {
    df[i, c("Cost", "Low", "High", "EGap", "EHigh", "ELow")] <- df[idx, c("Cost", "Low", "High", "EGap", "EHigh", "ELow")]
  } else {
    idx <- i
  }
}

# Filter data
df <- df[df$Endog. != "[0.8, 1.4, 1.9, 2.3, 1.7, 1.2, 0.8]\n[-0.48, -0.64, -0.8, -0.96, -1.12, -1.28, -1.44]", ]

# Function for integer breaks
integer_breaks <- function(n = 5) {
  breaker <- pretty_breaks(n)
  function(x) {
    breaks <- breaker(x)
    breaks[breaks == floor(breaks)]
  }
}

# Plot
ggplot(df[df$Iteration < 15, ], aes(x = Iteration, y = EGap)) +
  geom_line(color = "black", size = 1.5) +
  geom_point(color = "black", size = 3) +
  geom_ribbon(aes(ymin = ELow, ymax = EHigh), alpha = 0.2) +
  facet_grid(CS ~ Endog., scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 6, 1)) +
  scale_y_continuous(breaks = integer_breaks(), limits = c(-4, 33)) +
  theme_bw(base_size = 30) +
  theme(
    axis.line = element_line(color = "black"),
    legend.position = "top",
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")
  ) +
  labs(y = "Relative reduction in cost (%)", x = "Iteration")

# Save the plot as a PDF
ggsave("Figure 8.pdf", width = 20, height = 18, units = "in")