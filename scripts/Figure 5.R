# Load required libraries
library(ggplot2)
library(plyr)

# ####################### Data Preprocessing #######################

# Load dataset
df <- read.csv('ADPerformanceOptGap_Choice234.csv')

# Define and order levels for Cost Structure (CS)
df$CS <- factor(df$CS, levels = c("[10, 1, 20, 80]"))

# Define and order levels for Endogeneity (Endog.)
df$Endog. <- factor(df$Endog., levels = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]"))

# ####################### Update Optimality Gap #######################

# Track the best performance so far across iterations
for (i in 1:nrow(df)) {
  if (df$Iteration[i] == 0) {
    idx <- i  # Initialize tracking index at the start of each iteration
  }
  if (df$OptGap[i] > df$OptGap[idx]) {
    # Update values to the best observed so far
    df$OptGap[i] <- df$OptGap[idx]
    df$Cost[i] <- df$Cost[idx]
    df$Low[i] <- df$Low[idx]
    df$High[i] <- df$High[idx]
  } else {
    idx <- i  # Update index to the current row
  }
}

# ####################### Map Factors for Plot Labels #######################

# Map Endogeneity (Endog.) and Cost Structure (CS) to user-friendly names
df$Endog. <- mapvalues(df$Endog., 
                       from = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]"), 
                       to = c("(1, 0.5, 0.4, 0.8)", "(1, 0.5, 0.2, 0.4)"))
df$CS <- mapvalues(df$CS, 
                   from = c("[10, 1, 20, 80]"), 
                   to = c("(10, 1, 20, 80)"))

# ####################### Create and Save Plot #######################

# Plot Optimality Gap over iterations with facets for CS and Endog.
ggplot(df[df$Iteration < 17, ]) + 
  geom_line(aes(x = Iteration, y = OptGap, color = BasisFunction)) +  # Line plot
  geom_point(aes(x = Iteration, y = OptGap, color = BasisFunction, shape = BasisFunction)) +  # Points
  facet_grid(CS ~ Endog.) +  # Facets for CS and Endog.
  scale_x_continuous(breaks = seq(0, 16, 2)) +  # X-axis breaks
  scale_y_continuous(breaks = seq(0, 75, 10)) +  # Y-axis breaks
  theme_bw(base_size = 10) +  # Theme with base font size
  theme(
    legend.position = "top",
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +  # Format legend
  labs(y = 'Optimality Gap (%)', x = 'Iteration')  # Axis labels

# Save the plot as a PDF with the original size (5:3.2 inches)
ggsave("Figure 5.pdf", width = 5, height = 3.2, units = "in")
