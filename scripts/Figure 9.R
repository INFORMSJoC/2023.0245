# Load required libraries
library(ggplot2) # For plotting
library(plyr)    # For data manipulation
library(grid)    # For custom graphical elements
library(gridExtra) # For arranging multiple plots
library(cowplot) # For combining multiple ggplot objects

# Define color palette for the plots
colors <- c("Exogenous-Empirical" = "blue", 
            "Deterministic" = "red", 
            "Exogenous-ADP" = "green", 
            "Exogenous-Myopic" = "orange")

# Load the first dataset
df <- read.csv('ImpactEndog_m5.csv')

# Factorize columns to ensure proper ordering in plots
df$Cost <- factor(df$Cost, levels = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]",
                                      "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"))
df$Endogeneity <- factor(df$Endogeneity, levels = c(
  "[1.6, 2.6, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0]", 
  "[1.9, 3.1, 3.1, 2.5, 0.0, 0.0, 0.0, 0.0]",
  "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]",
  "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.08, -0.09]",
  "[1.9, 3.1, 3.1, 2.5, -0.05, -0.1, -0.15, -0.2]",
  "[1.9, 3.1, 3.1, 2.5, -0.1, -0.2, -0.3, -0.4]"
))

# Map endogeneity values to concise labels for readability
df$Endogeneity <- mapvalues(df$Endogeneity, 
                            from = c("[1.6, 2.6, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0]", 
                                     "[1.9, 3.1, 3.1, 2.5, 0.0, 0.0, 0.0, 0.0]",
                                     "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]",
                                     "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.08, -0.09]",
                                     "[1.9, 3.1, 3.1, 2.5, -0.05, -0.1, -0.15, -0.2]",
                                     "[1.9, 3.1, 3.1, 2.5, -0.1, -0.2, -0.3, -0.4]"), 
                            to = c("(1.6, 2.6, 2.8, 1.6)", 
                                   "(1.9, 3.1, 3.1, 2.5)\n(0.0, 0.0, 0.0, 0.0)", 
                                   "(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)", 
                                   "(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.08, -0.09)",
                                   "(1.9, 3.1, 3.1, 2.5)\n(-0.05, -0.1, -0.15, -0.2)", 
                                   "(1.9, 3.1, 3.1, 2.5)\n(-0.1, -0.2, -0.3, -0.4)"))

# Create a mapping of 'Cost' levels to numeric values for x-axis
strings <- sort(unique(df$Cost))
x <- 1:length(strings)
names(x) <- strings
df$x <- x[df$Cost]

# Filter data for deterministic case
df1 <- df[c(1, 2, 6:12)] # Select relevant columns
df1 <- df1[df1$Endogeneity != "(1.9, 3.1, 3.1, 2.5)\n(0.0, 0.0, 0.0, 0.0)", ]
df1$ShelfLife <- 'Deterministic' # Label for deterministic case
df1$scale <- 50 # Default scale for deterministic case

# Function to add tags to facets
tag_facet2 <- function(p, open = "(", close = ")", 
                       tag_pool = c(paste("D", c(1:2), sep=""), paste("E", c(1:4), sep="")), 
                       x = -Inf, y = Inf, hjust = -0.5, vjust = 1.5, 
                       fontface = 2, family = "", ...) {
  gb <- ggplot_build(p)
  lay <- gb$layout$layout
  tags <- cbind(lay, label = paste0(open, tag_pool[lay$PANEL], close), x = x, y = y)
  p + geom_text(data = tags, size=10, aes_string(x = "x", y = "y", label = "label"), 
                ..., hjust = hjust, vjust = vjust, fontface = fontface, family = family, inherit.aes = FALSE)
}

# Create the first plot for deterministic case
g1 <- ggplot(data=df1[which(df1$Endogeneity%in% c("(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)", 
                                                  "(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.08, -0.09)")),])+
  geom_line(aes(x =x , y=Det_Gap, color="Deterministic"), size = 1) +
  geom_point(aes(x =x , y=Det_Gap, color="Deterministic"), size = 2) +
  geom_ribbon(aes(x=x, ymin = DLow, ymax = DHigh), alpha=0.2, fill = 'red') +
  geom_point(aes(x = x, y = scale), colour = "white", size = 5) + 
  geom_line(aes(x = x, y = scale), colour = "white", size = 5) + 
  facet_grid(.~Endogeneity, scales = "free_y") + 
  scale_x_discrete(limits = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"),
                   labels = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"))+
  theme_bw(base_size = 30) +
  theme(legend.position = "top", 
        plot.background = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.background = element_rect(size=0.5, linetype="solid", colour ="black")) +
  guides(color=guide_legend(nrow=1, byrow=TRUE)) +
  labs(y='', x='')+
  scale_color_manual(values = colors)

gf1 = tag_facet2(g1)

# Create the second plot for deterministic case
df1$scale = 150

tag_facet2 <- function(p, open = "(", close = ")", tag_pool = c(paste("D", c(3:4), sep=""), paste("E", c(1:4), sep="")), x = -Inf, y = Inf, 
                       hjust = -0.5, vjust = 1.5, fontface = 2, family = "", ...) {
  
  gb <- ggplot_build(p)
  lay <- gb$layout$layout
  tags <- cbind(lay, label = paste0(open, tag_pool[lay$PANEL], close), x = x, y = y)
  p + geom_text(data = tags, size=10, aes_string(x = "x", y = "y", label = "label"), ..., hjust = hjust, 
                vjust = vjust, fontface = fontface, family = family, inherit.aes = FALSE)
}

g2 <- ggplot(data=df1[which(df1$Endogeneity%in% c("(1.9, 3.1, 3.1, 2.5)\n(-0.05, -0.1, -0.15, -0.2)", "(1.9, 3.1, 3.1, 2.5)\n(-0.1, -0.2, -0.3, -0.4)")),])+
  geom_line(aes(x =x , y=Det_Gap, color="Deterministic"), size = 1) +
  geom_point(aes(x =x , y=Det_Gap, color="Deterministic"), size = 2) +
  geom_ribbon(aes(x=x, ymin = DLow, ymax = DHigh), alpha=0.2, fill = 'red') +
  geom_point(aes(x = x, y = scale), colour = "white", size = 5) + #Trick
  geom_line(aes(x = x, y = scale), colour = "white", size = 5) + #Trick
  facet_grid(.~Endogeneity, scales = "free_y") + 
  scale_x_discrete(limits = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"),
                   labels = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"))+
  theme_bw(base_size = 30) +
  theme(legend.position = "none", 
        plot.background = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 1.05, hjust=1),
        legend.background = element_rect(size=0.5, linetype="solid", colour ="black")) +
  guides(color=guide_legend(nrow=1, byrow=TRUE)) +
  labs(y='', x='')+
  scale_color_manual(values = colors)

gf2 = tag_facet2(g2)

# Combine plots using cowplot
ggList = list(gf1, gf2)
plot <- cowplot::plot_grid(plotlist = ggList, rel_heights = c(1,1.3), ncol = 1, align = 'v')

# Load the second dataset
df <- read.csv('ImpactEndog_m5_ExgMypADP.csv')

df$Cost <- factor(df$Cost, levels = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"))
df$Endogeneity = factor(df$Endogeneity, levels = c("[1.6, 2.6, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0]", "[1.9, 3.1, 3.1, 2.5, 0.0, 0.0, 0.0, 0.0]",
                                                   "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]", "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.08, -0.09]",
                                                   "[1.9, 3.1, 3.1, 2.5, -0.05, -0.1, -0.15, -0.2]", "[1.9, 3.1, 3.1, 2.5, -0.1, -0.2, -0.3, -0.4]"))
df$Endogeneity = mapvalues(df$Endogeneity, from = c("[1.6, 2.6, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0]", "[1.9, 3.1, 3.1, 2.5, 0.0, 0.0, 0.0, 0.0]",
                                                    "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.03, -0.09]", "[1.9, 3.1, 3.1, 2.5, -0.03, -0.06, -0.08, -0.09]",
                                                    "[1.9, 3.1, 3.1, 2.5, -0.05, -0.1, -0.15, -0.2]", "[1.9, 3.1, 3.1, 2.5, -0.1, -0.2, -0.3, -0.4]"), 
                           to = c("(1.6, 2.6, 2.8, 1.6)", "(1.9, 3.1, 3.1, 2.5)\n(0.0, 0.0, 0.0, 0.0)", 
                                  "(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)", "(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.08, -0.09)",
                                  "(1.9, 3.1, 3.1, 2.5)\n(-0.05, -0.1, -0.15, -0.2)", "(1.9, 3.1, 3.1, 2.5)\n(-0.1, -0.2, -0.3, -0.4)"))

strings=sort(unique(df$Cost))
x = 1:length(strings)
names(x)=strings
df$x = x[df$Cost]


df2 = df[c(1,2,6:12)]
df2 = df2[which(df2$Endogeneity != "(1.9, 3.1, 3.1, 2.5)\n(0.0, 0.0, 0.0, 0.0)"), ]
df2 = cbind(df2, df1[c(3,4,5)])
df2$ShelfLife = 'Exogenous'
df2$scale = 20
df2$Exog_Gap[which(df2$Endogeneity!="(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)")] =20
df2$EHigh[which(df2$Endogeneity!="(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)")] =20
df2$ELow[which(df2$Endogeneity!="(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)")] =20

tag_facet2 <- function(p, open = "(", close = ")", tag_pool = c(paste("E", c(1:2), sep=""), paste("E", c(1:4), sep="")), x = -Inf, y = Inf, 
                       hjust = -0.5, vjust = 1.5, fontface = 2, family = "", ...) {
  
  gb <- ggplot_build(p)
  lay <- gb$layout$layout
  tags <- cbind(lay, label = paste0(open, tag_pool[lay$PANEL], close), x = x, y = y)
  p + geom_text(data = tags, size=10, aes_string(x = "x", y = "y", label = "label"), ..., hjust = hjust, 
                vjust = vjust, fontface = fontface, family = family, inherit.aes = FALSE)
}

# Create the first plot for stochastic case
g1 <- ggplot(data=df2[which(df2$Endogeneity%in% c("(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)", "(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.08, -0.09)")),])+
  geom_line(aes(x =x , y=ExogMyp_Gap, color="Exogenous-Myopic"), size = 1) +
  geom_point(aes(x =x , y=ExogMyp_Gap, color="Exogenous-Myopic"), size = 2) +
  geom_ribbon(aes(x=x, ymin = EMypLow, ymax = EMypHigh), alpha=0.2, fill = 'orange') +
  geom_line(aes(x =x , y=ExogADP_Gap, color="Exogenous-ADP"), size = 1) +
  geom_point(aes(x =x , y=ExogADP_Gap, color="Exogenous-ADP"), size = 2) +
  geom_ribbon(aes(x=x, ymin = EADPLow, ymax = EADPHigh), alpha=0.2, fill = 'green') +
  geom_line(aes(x =x , y=Exog_Gap, color="Exogenous-Empirical"), size = 1) +
  geom_point(aes(x =x , y=Exog_Gap, color="Exogenous-Empirical"), size = 2) +
  geom_ribbon(aes(x=x, ymin = ELow, ymax = EHigh), alpha=0.2, fill = 'blue') +
  geom_point(aes(x = x, y = scale), colour = "white", size = 5) + #Trick
  geom_line(aes(x = x, y = scale), colour = "white", size = 5) + #Trick
  facet_grid(.~Endogeneity, scales = "free_y") + #, ncol = 6, nrow = 1
  scale_x_discrete(limits = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"),
                   labels = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"))+
  theme_bw(base_size = 30) +
  theme(legend.position = "top", 
        plot.background = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.background = element_rect(size=0.5, linetype="solid", colour ="black")) +
  guides(color=guide_legend(nrow=1, byrow=TRUE)) +
  labs(y='', x='')+
  scale_color_manual(values = colors)

gf3 = tag_facet2(g1)

# Create the second plot for stochastic case
df2$scale = 84
df2$Exog_Gap[which(df2$Endogeneity!="(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)")] =84
df2$EHigh[which(df2$Endogeneity!="(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)")] =84
df2$ELow[which(df2$Endogeneity!="(1.9, 3.1, 3.1, 2.5)\n(-0.03, -0.06, -0.03, -0.09)")] =84

tag_facet2 <- function(p, open = "(", close = ")", tag_pool = c(paste("E", c(3:4), sep=""), paste("E", c(1:4), sep="")), x = -Inf, y = Inf, 
                       hjust = -0.5, vjust = 1.5, fontface = 2, family = "", ...) {
  
  gb <- ggplot_build(p)
  lay <- gb$layout$layout
  tags <- cbind(lay, label = paste0(open, tag_pool[lay$PANEL], close), x = x, y = y)
  p + geom_text(data = tags, size=10, aes_string(x = "x", y = "y", label = "label"), ..., hjust = hjust, 
                vjust = vjust, fontface = fontface, family = family, inherit.aes = FALSE)
}

g2 <- ggplot(data=df2[which(df2$Endogeneity%in% c("(1.9, 3.1, 3.1, 2.5)\n(-0.05, -0.1, -0.15, -0.2)",
                                                  "(1.9, 3.1, 3.1, 2.5)\n(-0.1, -0.2, -0.3, -0.4)")),])+# & df6$ShelfLife =="Exogenous"),]) +
  geom_line(aes(x =x , y=ExogMyp_Gap, color="Exogenous-Myopic"), size = 1) +
  geom_point(aes(x =x , y=ExogMyp_Gap, color="Exogenous-Myopic"), size = 2) +
  geom_ribbon(aes(x=x, ymin = EMypLow, ymax = EMypHigh), alpha=0.2, fill = 'orange') +
  geom_line(aes(x =x , y=ExogADP_Gap, color="Exogenous-ADP"), size = 1) +
  geom_point(aes(x =x , y=ExogADP_Gap, color="Exogenous-ADP"), size = 2) +
  geom_ribbon(aes(x=x, ymin = EADPLow, ymax = EADPHigh), alpha=0.2, fill = 'green') +
  geom_line(aes(x =x , y=Exog_Gap, color="Exogenous-Empirical"), size = 1) +
  geom_point(aes(x =x , y=Exog_Gap, color="Exogenous-Empirical"), size = 2) +
  geom_ribbon(aes(x=x, ymin = ELow, ymax = EHigh), alpha=0.2, fill = 'blue') +
  geom_point(aes(x = x, y = scale), colour = "white", size = 5) + #Trick
  geom_line(aes(x = x, y = scale), colour = "white", size = 5) + #Trick
  facet_grid(.~Endogeneity, scales = "free_y") + #, ncol = 6, nrow = 1
  scale_x_discrete(limits = c("[10, 1, 20, 5]", "[10, 1, 20, 20]", "[10, 1, 20, 80]", "[100, 1, 20, 5]", "[100, 1, 20, 20]", "[100, 1, 20, 80]"),
                   labels = c("(10, 1, 20, 5)", "(10, 1, 20, 20)", "(10, 1, 20, 80)", "(100, 1, 20, 5)", "(100, 1, 20, 20)", "(100, 1, 20, 80)"))+
  theme_bw(base_size = 30) +
  theme(legend.position = "none", 
        plot.background = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 1.05, hjust=1),
        # axis.title.x=element_blank(),
        # axis.text.x=element_blank(),
        # axis.ticks.x=element_blank(),
        legend.background = element_rect(size=0.5, linetype="solid", colour ="black")) +
  guides(color=guide_legend(nrow=1, byrow=TRUE)) +
  labs(y='', x='')+
  scale_color_manual(values = colors)

gf4 = tag_facet2(g2)

# Combine plots using cowplot
ggList1 = list(gf3, gf4)
plot1 <- cowplot::plot_grid(plotlist = ggList1, rel_heights = c(1,1.3), ncol = 1, align = 'v')

# Combine plots using cowplot
ggList2 = list(plot, plot1)
plot2 <- cowplot::plot_grid(plotlist = ggList2, rel_heights = c(1,1.3), ncol = 2, align = 'v')


y.grob <- textGrob("Relative Gap (%)", 
                   gp=gpar(col="black", fontsize=30), rot=90)

x.grob <- textGrob('(Fixed ordering cost, Unit holding cost, Unit shortage cost, Unit wastage cost)', 
                   gp=gpar(col="black", fontsize=30))


tagged_plot <- grid.arrange(arrangeGrob(plot2, left = y.grob, bottom = x.grob))

# Save the plot as a PDF
ggsave("Figure 10.pdf", plot = tagged_plot, width = 26, height = 16, units = "in")
