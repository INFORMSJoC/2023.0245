# Load required libraries
library(ggplot2)
library(plyr)

# ####################### Data Preprocessing #######################

# Load dataset
df <- read.csv('LBA_InfoRelaxRandHorizonLength_4000rep_v2.csv')

# Define and order levels for Cost
df$Cost <- factor(df$Cost, levels = c(
  "[10, 1, 20, 5]", 
  "[10, 1, 20, 20]", 
  "[10, 1, 20, 80]", 
  "[100, 1, 20, 5]", 
  "[100, 1, 20, 20]", 
  "[100, 1, 20, 80]"
))

# Define and order levels for Endogeneity
df$Endogeneity <- factor(df$Endogeneity, levels = c(
  "[1, 0.5, 0.4, 0.8]", 
  "[1, 0.5, 0.2, 0.4]", 
  "[1, 0.5, 0.0, 0.0]", 
  "[1, 0.5, -0.1, -0.05]", 
  "[1, 0.5, -0.2, -0.1]", 
  "[1, 0.5, -0.4, -0.8]"
))

# Map numeric indices for unique Cost levels
unique_costs <- sort(unique(df$Cost))
cost_indices <- 1:length(unique_costs)
names(cost_indices) <- unique_costs
df$x <- cost_indices[df$Cost]

# Calculate Relative Gap
df$Gap <- (df$Optimal - df$InfoRelax) * 100 / df$Optimal

# ####################### Data Transformation #######################

# Create two datasets for combining into a single dataframe later
# Dataset 1: Cost perspective
df1 <- df
df1$EGapLow <- df1$InfoRelax
df1$EGapHigh <- df1$InfoRelax
df1$Criteria <- 'Cost'

# Dataset 2: Relative Gap perspective
df2 <- df
df2$InfoRelax <- (df2$Optimal - df2$InfoRelax) * 100 / df2$Optimal
df2$Low <- df2$InfoRelax
df2$High <- df2$InfoRelax
df2$Optimal <- df2$InfoRelax
df2$Criteria <- 'Relative Gap (%)'

# Combine datasets
df3 <- rbind(df1, df2)

# ####################### Map Factors for Plot Labels #######################

# Update labels for Endogeneity
df3$Endogeneity <- mapvalues(df3$Endogeneity, 
                             from = c(
                               "[1, 0.5, 0.4, 0.8]", 
                               "[1, 0.5, 0.2, 0.4]", 
                               "[1, 0.5, 0.0, 0.0]", 
                               "[1, 0.5, -0.1, -0.05]", 
                               "[1, 0.5, -0.2, -0.1]", 
                               "[1, 0.5, -0.4, -0.8]"), 
                             to = c(
                               "(1, 0.5, 0.4, 0.8)", 
                               "(1, 0.5, 0.2, 0.4)", 
                               "(1, 0.5, 0.0, 0.0)", 
                               "(1, 0.5, -0.1, -0.05)", 
                               "(1, 0.5, -0.2, -0.1)", 
                               "(1, 0.5, -0.4, -0.8)"))

# ####################### Create and Save Plot #######################

# Define custom colors for lines
colors <- c("Optimal" = "blue", "Lower-Bound" = "red")

# Generate the plot
ggplot(data = df3) + 
  geom_line(aes(x = x, y = Optimal, color = 'Optimal'), size = 1, linetype = "solid") +  # Line for Optimal
  geom_point(aes(x = x, y = Optimal, color = 'Optimal'), size = 4) +  # Points for Optimal
  geom_line(aes(x = x, y = InfoRelax, color = 'Lower-Bound'), size = 1, linetype = "solid") +  # Line for Lower-Bound
  geom_point(aes(x = x, y = InfoRelax, color = 'Lower-Bound'), size = 4) +  # Points for Lower-Bound
  geom_ribbon(aes(x = x, ymin = Low, ymax = High), alpha = 0.2, fill = 'red') +  # Ribbon for Low-High range
  geom_ribbon(aes(x = x, ymin = EGapLow, ymax = EGapHigh), alpha = 0.2, linetype = 0, fill = "red", col = 'red') +  # Ribbon for EGapLow-EGapHigh
  facet_grid(Criteria ~ Endogeneity, scales = 'free_y') +  # Facet by Criteria and Endogeneity
  scale_x_discrete(
    limits = c(
      "[10, 1, 20, 5]", 
      "[10, 1, 20, 20]", 
      "[10, 1, 20, 80]", 
      "[100, 1, 20, 5]", 
      "[100, 1, 20, 20]", 
      "[100, 1, 20, 80]"
    ),
    labels = c(
      "(10, 1, 20, 5)", 
      "(10, 1, 20, 20)", 
      "(10, 1, 20, 80)", 
      "(100, 1, 20, 5)", 
      "(100, 1, 20, 20)", 
      "(100, 1, 20, 80)"
    )
  ) +
  theme_bw(base_size = 25) +  # Base theme with font size
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1.05, hjust = 1),  # Rotate x-axis text
    legend.position = "top",
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")
  ) +
  labs(y = '', x = '(Fixed ordering costs, Unit cost of holding, Unit cost of shortage, Unit cost of wastage)') +
  scale_color_manual(values = colors)  # Custom line colors

# Save the plot as a PDF with the original size (20:12 inches)
ggsave("Figure 6.pdf", width = 20, height = 12, units = "in")
