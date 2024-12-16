# Load necessary libraries
library(ggplot2)
library(plyr)
library(scales)

###### Demand Sensitivity Analysis #####

# Load and process data for M=20
df <- read.csv('ADPerformance_Myopic_Exact_RelGap.csv')

# Factorize cost structures (CS) and endogeneity levels (Endog.)
df$CS <- factor(df$CS, levels = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"))
df$Endog. <- factor(df$Endog., levels = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]", "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"))

# Create a copy of the dataset for transformation
df1 <- df

# Adjust rows based on iteration and minimum EGap value
df1$RelGap0 <- NA
for (i in 1:nrow(df1)) {
  if (df1$Iteration[i] == 0) {
    idx <- i
  }
  if (df1$EGap[i] < df1$EGap[idx]) {
    df1$Cost[i] <- df1$Cost[idx]
    df1$Low[i] <- df1$Low[idx]
    df1$High[i] <- df1$High[idx]
    df1$EGap[i] <- df1$EGap[idx]
    df1$EHigh[i] <- df1$EHigh[idx]
    df1$ELow[i] <- df1$ELow[idx]
  } else {
    idx <- i
  }
}

df2 <- df1

# Load and process data for M=30
df <- read.csv('ADPerformance_Myopic_Exact_DemSens_v1.csv')

# Factorize columns for consistency
df$CS <- factor(df$CS, levels = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"))
df$Endog. <- factor(df$Endog., levels = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]", "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"))

# Create a copy for transformation
df1 <- df

# Adjust rows as done for M=20
df1$RelGap0 <- NA
for (i in 1:nrow(df1)) {
  if (df1$Iteration[i] == 0) {
    idx <- i
  }
  if (df1$EGap[i] < df1$EGap[idx]) {
    df1$Cost[i] <- df1$Cost[idx]
    df1$Low[i] <- df1$Low[idx]
    df1$High[i] <- df1$High[idx]
    df1$EGap[i] <- df1$EGap[idx]
    df1$EHigh[i] <- df1$EHigh[idx]
    df1$ELow[i] <- df1$ELow[idx]
  } else {
    idx <- i
  }
}

# Select relevant columns and rename them
df1 <- df1[4:9]
colnames(df1) <- c("Cost1", "Low1", "High1", "EGap1", "ELow1", "EHigh1")

# Combine M=20 and M=30 data
df3 <- cbind(df2, df1)

# Map old values to cleaner labels for Endog. and CS
df3$Endog. <- mapvalues(df3$Endog., from = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, 0.0, 0.0]", "[1, 0.5, -0.1, -0.05]", "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.4, -0.8]"), 
                        to = c("(1, 0.5, 0.4, 0.8)", "(1, 0.5, 0.2, 0.4)", "(1, 0.5, 0.0, 0.0)", "(1, 0.5, -0.1, -0.05)", "(1, 0.5, -0.2, -0.1)", "(1, 0.5, -0.4, -0.8)"))
df3$CS <- mapvalues(df3$CS, from = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"),
                    to = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"))

# Add scaling values
df3$scale <- c(rep(5, 6 * 21), rep(5, 6 * 21), rep(36, 6 * 21), rep(36, 6 * 21), rep(23, 6 * 21), rep(10, 6 * 21))

# Define color scheme
colors <- c("M=30" = "blue", "M=20" = "red")

# Define a function for integer breaks in y-axis
integer_breaks <- function(n = 5, ...) {
  breaker <- pretty_breaks(n, ...)
  function(x) {
    breaks <- breaker(x)
    breaks[breaks == floor(breaks)]
  }
}

# Plot the data
ggplot(data = df3[which(df3$Iteration < 15), ]) +
  geom_line(aes(x = Iteration, y = EGap, color = 'M=20'), size = 1, linetype = "solid") +
  geom_point(aes(x = Iteration, y = EGap, color = 'M=20'), size = 2, shape = 19) +
  geom_ribbon(aes(x = Iteration, ymin = ELow, ymax = EHigh), alpha = 0.2, linetype = 0, fill = "red", col = 'red') +
  geom_line(aes(x = Iteration, y = EGap1, color = 'M=30'), size = 1, linetype = "solid") +
  geom_point(aes(x = Iteration, y = EGap1, color = 'M=30'), size = 2, shape = 19) +
  geom_ribbon(aes(x = Iteration, ymin = ELow1, ymax = EHigh1), alpha = 0.2, linetype = 0, fill = "blue", col = 'blue') +
  geom_point(aes(x = Iteration, y = scale), colour = "white", size = 0.1) + 
  facet_grid(CS ~ Endog., scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 12, 4)) +
  scale_y_continuous(breaks = integer_breaks()) +
  theme_bw(base_size = 30) +
  theme(axis.line = element_line(color = 'black'), legend.position = "top",
        panel.grid.minor = element_blank(), panel.grid.major = element_blank(),
        legend.title = element_blank(), legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")) +
  labs(y = 'Relative reduction in cost (%)', x = 'Iteration') +
  scale_color_manual(values = colors)

# Save the plot as a PDF
ggsave("Figure 13.pdf", width = 22, height = 19, units = "in")
