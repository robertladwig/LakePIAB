#setwd("C:/Users/ladwi/Documents/Projects/R/LakePIAB")
setwd("/Users/robertladwig/Documents/DSI/LakePIAB")
# setwd('/home/robert/Projects/LakePIAB')
library(tidyverse)
library(rLakeAnalyzer)
library(patchwork)
library(lubridate)
library(RColorBrewer)

pa1 <- read.csv("stability/parallel_init.csv")
pa2 <- read.csv("stability/parallel_heat.csv")
pa3 <- read.csv("stability/parallel_ice.csv")
pa4 <- read.csv("stability/parallel_diff.csv")
pa5 <- read.csv("stability/parallel_conv.csv")
pa6 <- read.csv("stability/parallel_init.csv")

si1 <- read.csv("stability/single_init.csv")
si2 <- read.csv("stability/single_heat.csv")
si3 <- read.csv("stability/single_ice.csv")
si4 <- read.csv("stability/single_diff.csv")
si5 <- read.csv("stability/single_conv.csv")
si6 <- read.csv("stability/single_init.csv")

pb1 <- read.csv("stability/py_temp_initial00.csv")
pb2 <- read.csv("stability/py_temp_heat01.csv")
pb3 <- read.csv("stability/py_temp_total05.csv")
pb4 <- read.csv("stability/py_temp_diff02.csv")
pb5 <- read.csv("stability/py_temp_conv04.csv")


obs <- read.csv("stability/observed.csv")


heat <-  sqrt((pa2[,2:ncol(pa1)] - si2[,2:ncol(pa1)])**2) %>% mutate(time = as.POSIXct(pa2$time))
ice <-  sqrt((pa3[,2:ncol(pa1)] - si3[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
diff <-  sqrt((pa4[,2:ncol(pa1)] - si4[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
conv <-  sqrt((pa5[,2:ncol(pa1)] - si5[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))

heat_p <-  sqrt((pa2[,2:ncol(pa1)] - pb2[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
ice_p <-  sqrt((pa3[,2:ncol(pa1)] - pb3[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
diff_p <-  sqrt((pa4[,2:ncol(pa1)] - pb4[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
conv_p <-  sqrt((pa5[,2:ncol(pa1)] - pb5[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))

heat_s <-  sqrt((si2[,2:ncol(pa1)] - pb2[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
ice_s <-  sqrt((si3[,2:ncol(pa1)] - pb3[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
diff_s <-  sqrt((si4[,2:ncol(pa1)] - pb4[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))
conv_s <-  sqrt((si5[,2:ncol(pa1)] - pb5[,2:ncol(pa1)])**2)%>% mutate(time = as.POSIXct(pa2$time))


ggplot() +
  geom_line(data = heat, aes(seq(1, length(heat[,1])), heat[,1], col = 'ice')) + 
  geom_line(data = ice, aes(seq(1, length(ice[,1])), ice[,1], col = 'diff')) + 
  geom_line(data = diff, aes(seq(1, length(diff[,1])), diff[,1], col = 'conv')) + 
  geom_line(data = conv, aes(seq(1, length(conv[,1])), conv[,1], col = 'final')) +
  scale_y_continuous(trans = 'log10')
  xlim(0, 700)+ylim(-100,3)
            
  
g_heat <- ggplot() +
  geom_line(data = heat_p, aes(time, X0,  col = "Parallel - Process")) +
  geom_line(data = heat_s, aes(time, X0,  col = "Single - Process")) +
  # geom_line(data = heat, aes(time, X0, col = "Parallel - Single")) +
  scale_y_continuous(trans = 'log10') +
  ylab(expression("log10" *Delta * "Temperature")) +
  xlab("Time step (3600 s)") +
  scale_colour_manual(values=c('blue','red', 'red')) +
  ggtitle('Output after heating step') +
  theme_bw() +
  theme(legend.title = element_blank()) ; g_heat

g_ice <- ggplot() +
  geom_line(data =ice_p, aes(time, X0,  col = "Parallel - Process")) +
  geom_line(data = ice_s, aes(time, X0,  col = "Single - Process")) +
  # geom_line(data = ice, aes(time, X0, col = "Parallel - Single")) +
  scale_y_continuous(trans = 'log10') +
  ylab(expression("log10" *Delta * "Temperature")) +
  xlab("Time step (3600 s)") +
  scale_colour_manual(values=c('blue','red', 'red')) +
  ggtitle('Output after ice step') +
  theme_bw() +
  theme(legend.title = element_blank()) ; g_ice

g_diff <- ggplot() +
  geom_line(data = diff_p, aes(time, X0,  col = "Parallel - Process")) +
  geom_line(data = diff_s, aes(time, X0,  col = "Single - Process")) +
  # geom_line(data = diff, aes(time, X0, col = "Parallel - Single")) +
  scale_y_continuous(trans = 'log10') +
  ylab(expression("log10" *Delta * "Temperature")) +
  xlab("Time step (3600 s)") +
  scale_colour_manual(values=c('blue','red', 'red')) +
  ggtitle('Output after diffusion (DL) step') +
  theme_bw() +
  theme(legend.title = element_blank()) ; g_diff

g_conv <- ggplot() +
  geom_line(data = conv_p, aes(time, X0,  col = "Parallel - Process")) +
  geom_line(data = conv_s, aes(time, X0,  col = "Single - Process")) +
  # geom_line(data = conv, aes(time, X0, col = "Parallel - Single")) +
  scale_y_continuous(trans = 'log10') +
  ylab(expression("log10" *Delta * "Temperature")) +
  xlab("Time step (3600 s)") +
  scale_colour_manual(values=c('blue','red', 'red')) +
  ggtitle('Output after convection step') +
  theme_bw() +
  theme(legend.title = element_blank()) ; g_conv

g <-ggplot() +
  geom_line(data = conv_p, aes(seq(1,nrow(conv_p)), X0,  col = "Parallel - Process")) +
  geom_line(data = conv_s, aes(seq(1,nrow(conv_s)), X0,  col = "Single - Process")) +
  geom_point(data = conv_p, aes(seq(1,nrow(conv_p)), X0,  col = "Parallel - Process")) +
  geom_point(data = conv_s, aes(seq(1,nrow(conv_s)), X0,  col = "Single - Process")) +
  # geom_line(data = conv, aes(time, X0, col = "Parallel - Single")) +
  scale_y_continuous(trans = 'log10') +
  ylab(expression("log10(" *Delta * "Temperature (\u00B0C))")) +
  xlab("Time step (3600 s)") +
  scale_colour_manual(values=c('blue','red', 'red')) +
  ggtitle('Output after final hybrid MCL model step') +
  xlim(0,1000)+
  theme_bw() +
  theme(legend.title = element_blank()) ; g

p9 = g_heat / g_ice / g_diff / g_conv & plot_layout(guides = 'collect') & plot_annotation(tag_levels = 'A')# &theme(legend.position = 'bottom')
ggsave(plot = p9, filename = "figs/Fig7_stability.png", dpi = 300, width = 9, height =12, units = 'in')

ggsave(plot = g, filename = "figs/Fig8_stability.png", dpi = 300, width = 15, height =6, units = 'in')
