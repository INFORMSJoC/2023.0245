# Load required libraries
library(ggplot2)
library(plyr)

# ############### Initial Results ###############

# Load dataset for initial results
df <- read.csv('RMSE_BFA_Paper_LQC_Interaction_Revision.csv')

# Define and order levels for the Cost factor
df$Cost <- factor(df$Cost, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))

# Create a mapping for Cost strings to numeric values
strings <- sort(unique(df$Cost))
x <- 1:length(strings)
names(x) <- strings
df$x <- x[df$Cost]

# Define and order levels for the Endogeneity factor
df$Endogeneity <- factor(df$Endogeneity, levels = c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
  "[1, 0.5, -0.2, -0.4]", "[1, 0.5, -0.4, -0.8]"
))

# Define and order levels for the day factor
df$day <- factor(df$day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))

# Filter the dataframe for specific Endogeneity levels
df2 <- df[df$Endogeneity %in% c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
  "[1, 0.5, -0.4, -0.8]"
), ]

# ############### Extended Results ###############

# Load dataset for extended results
df <- read.csv('RMSE_BFA_Paper_LQC_Interaction_NegParams_Revision.csv')

# Redefine the Cost factor with ordering
df$Cost <- factor(df$Cost, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))

# Map Cost strings to numeric values
strings <- sort(unique(df$Cost))
x <- 1:length(strings)
names(x) <- strings
df$x <- x[df$Cost]

# Define and order levels for the Endogeneity factor
df$Endogeneity <- factor(df$Endogeneity, levels = c(
  "[1, 0.5, 0.0, 0.0]", "[1, 0.5, -0.05, -0.1]", "[1, 0.5, -0.1, -0.2]",
  "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]"
))

# Define the day factor
df$day <- factor(df$day, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))

# Filter the dataframe for specific Endogeneity levels
df3 <- df[df$Endogeneity %in% c("[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]"), ]

# ############### Final Combined Results ###############

# Combine filtered dataframes
df <- rbind(df3, df2)

# Re-define factor levels for consistency
df$Cost <- factor(df$Cost, levels = c(
  "[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
  "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"
))

df$Endogeneity <- factor(df$Endogeneity, levels = c(
  "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
  "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"
))

# ############### Prepare Data for Plotting ###############

# Map Approximation levels to user-friendly names
df$Approximation <- factor(df$Approximation, levels = c("x1", "x1^2", "x1^3", "x1x2"))
df$Approximation <- mapvalues(
  df$Approximation,
  from = c("x1", "x1^2", "x1^3", "x1x2"),
  to = c("Choice 1", "Choice 2", "Choice 3", "Choice 4")
)

# Filter data for specific days
df <- df[df$day %in% c("Mon", "Wed", "Fri", "Sat", "Sun"), ]

# Map Endogeneity levels to user-friendly names
df$Endogeneity <- mapvalues(
  df$Endogeneity,
  from = c(
    "[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]",
    "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"
  ),
  to = c(
    "(1, 0.5, 0.4, 0.8)", "(1, 0.5, 0.2, 0.4)", "(1, 0.5, 0.0, 0.0)",
    "(1, 0.5, -0.1, -0.05)", "(1, 0.5, -0.2, -0.1)", "(1, 0.5, -0.4, -0.8)"
  )
)

# ############### Create and Save Plot ###############

ggplot(data = df, aes(x = x, y = MAPE, colour = Approximation)) +
  geom_line(size = 1.5, linetype = "solid") +
  geom_point(aes(shape = Approximation), size = 4) +
  facet_grid(Endogeneity ~ day) +
  scale_x_discrete(
    limits = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"),
    labels = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)")
  ) +
  scale_y_continuous(breaks = seq(0, 10, 1)) +
  theme_bw(base_size = 35) +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1.05, hjust = 1),
    legend.position = "top",
    axis.line = element_line(color = 'black'),
    plot.background = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    legend.title = element_blank(),
    legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  labs(y = 'MAPE', x = '(Fixed ordering cost, Unit holding cost, Unit shortage cost, Unit wastage cost)')

# Save plot as PDF with custom size
ggsave("Figure 3.pdf", width = 28, height = 28)
