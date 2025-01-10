# Load required libraries
library(ggplot2)  
library(plyr)     
library(egg)      

# Define custom colors for the plot
colors <- c("Deterministic" = "red", "Exogenous-Optimal" = "green", "Exogenous-Myopic" = "orange")

# Load the data
df <- read.csv('Impact_ExogOptMyp_m3_v3_Cleaned.csv')  

# Convert 'Cost' to a factor and define the levels for ordering
df$Cost <- factor(df$Cost, levels = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", 
                                      "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"))

# Convert 'Endogeneity' to a factor and define the levels for ordering
df$Endogeneity <- factor(df$Endogeneity, levels = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", 
                                                    "[1, 0.5, -0.15, -0.05]", "[1, 0.5, -0.2, -0.1]", 
                                                    "[1, 0.5, -0.25, -0.15]", "[1, 0.5, -0.3, -0.2]", 
                                                    "[1, 0.5, -0.35, -0.25]", "[1, 0.5, -0.4, -0.3]"))

# Replace factor labels for 'Endogeneity' for cleaner display
df$Endogeneity <- mapvalues(df$Endogeneity, 
                            from = c("[1, 0.5, 0.4, 0.8]", "[1, 0.5, 0.2, 0.4]", "[1, 0.5, -0.15, -0.05]", 
                                     "[1, 0.5, -0.2, -0.1]", "[1, 0.5, -0.25, -0.15]", "[1, 0.5, -0.3, -0.2]", 
                                     "[1, 0.5, -0.35, -0.25]", "[1, 0.5, -0.4, -0.3]"), 
                            to = c("(1, 0.5, 0.4, 0.8)", "(1, 0.5, 0.2, 0.4)", "(1, 0.5, -0.15, -0.05)", 
                                   "(1, 0.5, -0.2, -0.1)", "(1, 0.5, -0.25, -0.15)", "(1, 0.5, -0.3, -0.2)", 
                                   "(1, 0.5, -0.35, -0.25)", "(1, 0.5, -0.4, -0.3)"))

# Define a custom function to add tags to facets
tag_facet2 <- function(p, open = "(", close = ")", 
                       tag_pool = c(paste("D", 1:8, sep=""), paste("E", 1:8, sep="")), 
                       x = -Inf, y = Inf, hjust = -0.5, vjust = 1.5, 
                       fontface = 2, family = "", ...) {
  gb <- ggplot_build(p)
  lay <- gb$layout$layout
  tags <- cbind(lay, label = paste0(open, tag_pool[lay$PANEL], close), x = x, y = y)
  p + geom_text(data = tags, size = 10, aes_string(x = "x", y = "y", label = "label"), 
                ..., hjust = hjust, vjust = vjust, fontface = fontface, family = family, inherit.aes = FALSE)
}

# Create the main plot
p <- ggplot(data = df) +
  geom_line(aes(x = x, y = ExogMyp_Gap, color = "Exogenous-Myopic"), size = 1) +
  geom_point(aes(x = x, y = ExogMyp_Gap, color = "Exogenous-Myopic"), size = 2) +
  geom_ribbon(aes(x = x, ymin = EMypLow, ymax = EMypHigh), alpha = 0.2, fill = 'orange') +
  geom_line(aes(x = x, y = ExogOpt_Gap, color = "Exogenous-Optimal"), size = 1) +
  geom_point(aes(x = x, y = ExogOpt_Gap, color = "Exogenous-Optimal"), size = 2) +
  geom_ribbon(aes(x = x, ymin = EOptLow, ymax = EOptHigh), alpha = 0.2, fill = 'green') +
  geom_line(aes(x = x, y = Det_OptGap, color = "Deterministic"), size = 1) +
  geom_point(aes(x = x, y = Det_OptGap, color = "Deterministic"), size = 2) +
  geom_point(aes(x = x, y = scale), colour = "white", size = 5) +  # Invisible trick
  geom_line(aes(x = x, y = scale), colour = "white", size = 5) +  # Invisible trick
  facet_grid(ShelfLife ~ Endogeneity, scales = "free_y") + 
  scale_x_discrete(limits = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", 
                              "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"), 
                   labels = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", 
                              "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)")) +
  theme_bw(base_size = 30) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1.05, hjust = 1), 
        legend.position = "top", plot.background = element_blank(), 
        panel.grid.minor = element_blank(), panel.grid.major = element_blank(), 
        legend.title = element_blank(), 
        legend.background = element_rect(size = 0.5, linetype = "solid", colour = "black")) +
  labs(y = 'Optimality Gap (%)', 
       x = '(Fixed ordering cost, Unit holding cost, Unit shortage cost, Unit wastage cost)') +
  scale_color_manual(values = colors)

# Add facet tags and save the plot
tagged_plot <- tag_facet2(p)
ggsave("Figure 8.pdf", plot = tagged_plot, width = 30, height = 15, units = "in")
