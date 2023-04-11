setwd("C:/Users/ladwi/Documents/Projects/R/LakePIAB")
# setwd("/Users/robertladwig/Documents/DSI/LakePIAB")
library(tidyverse)
library(rLakeAnalyzer)
library(patchwork)
library(lubridate)
library(RColorBrewer)

depths <- seq(0.5,24.5, 0.5)
depths <- c(0.05, depths)

hyps <- read.csv('input/bathymetry.csv')
area <- approx(hyps$Depth_meter, hyps$Area_meterSquared, depths)$y

physicalCalc <- function(input, area, depths, meteo, hyps){
  
  times <- input[,1]
  temp <- input[,-1]
  
  df = data.frame('time' = NULL,
                  'thermoclineDepth' = NULL,
                  'EpiDepth' = NULL,
                  'HypoDepth' = NULL,
                  'SchmidtStability' = NULL,
                  'LakeNumber' = NULL,
                  'Iso13' = NULL,
                  'Iso15' = NULL,
                  'Iso17' = NULL,
                  "SurfaceWTR" = NULL,
                  "BottomWTR" = NULL,
                  "N2" = NULL,
                  'EpiDense' = NULL,
                  'HypoDense' = NULL,
                  'MetaDense' = NULL)
  
  for (i in 1:nrow(pb)){
    
    print(round((100 * i) / nrow(pb)), 2)
    
    wtr <- as.numeric(temp[i,])
    wtr[wtr == -999] = NA
    
    TD <- thermo.depth(wtr = wtr,depths = depths)
    
    N2 <- max(buoyancy.freq(wtr = wtr,depths = depths))
    
    St <- schmidt.stability(wtr = wtr,
                            depths = depths,
                            bthA = hyps$Area_meterSquared,
                            bthD = hyps$Depth_meter)
    
    MetaDepths <- meta.depths(wtr = wtr,
                depths = depths, seasonal = T)
    
    if (any(is.na(MetaDepths))){
      EpiDense <- NA
      HypoDense <- NA
      MetaDense <- NA
      WindVel <- NA
      LN <- NA
    } else {
      
      EpiDense <- layer.density(top = 0, bottom = MetaDepths[1], wtr = wtr,
                                depths = depths, bthA = hyps$Area_meterSquared, bthD = hyps$Depth_meter)
      
      HypoDense <- layer.density(top = MetaDepths[2], bottom = max(depths), wtr = wtr,
                                 depths = depths, bthA = hyps$Area_meterSquared, bthD = hyps$Depth_meter)
      
      MetaDense <- layer.density(top = MetaDepths[1], bottom = MetaDepths[2], wtr = wtr,
                                 depths = depths, bthA = hyps$Area_meterSquared, bthD = hyps$Depth_meter)
      
      WindVel <- uStar(wndSpeed = meteo$Uw[i], wndHeight = 10, averageEpiDense = EpiDense)
      
      LN <- lake.number(bthA = hyps$Area_meterSquared, 
                        bthD = hyps$Depth_meter, 
                        uStar = WindVel, 
                        St = St, 
                        metaT = MetaDepths[1], 
                        metaB = MetaDepths[2], 
                        averageHypoDense = HypoDense)
      

    }
    
    SurfTemp <- wtr[2]
    
    BottomTemp <- wtr[length(wtr)-1]
    
    iso13 <- approx(x = wtr, y = depths, xout = 13)$y
    iso15 <- approx(x = wtr, y = depths, xout = 15)$y
    iso17 <- approx(x = wtr, y = depths, xout = 17)$y
    
    df = rbind(df, data.frame('time' = as.POSIXct(times[i]),
                    'thermoclineDepth' = TD,
                    'EpiDepth' = MetaDepths[1],
                    'HypoDepth' = MetaDepths[2],
                    'SchmidtStability' = St,
                    'LakeNumber' = LN,
                    'Iso13' = iso13,
                    'Iso15' = iso15,
                    'Iso17' = iso17,
                    "SurfaceWTR" = SurfTemp,
                    "BottomWTR" = BottomTemp,
                    'N2' = N2,
                    'EpiDense' = EpiDense,
                    'HypoDense' = HypoDense,
                    'MetaDense' = MetaDense))
  }
  
  return(df)
  
}

calc_rmse <- function(sim, obs){
  idx = !is.na(obs)
  rmse = sqrt(sum((sim[idx] - obs[idx])**2, na.rm = T) / length(obs[idx]))
  return(rmse)
}

calc_nse <- function(sim, obs){
  idx = !is.na(obs)
  rmse = 1- (sum((obs[idx] - sim[idx])**2, na.rm = T) /sum((obs[idx] - mean(obs[idx], na.rm = T))**2, na.rm = T))
  return(rmse)
}

pb <- read.csv("verification/pb_temp.csv")
pb_met <- read.csv("verification/pb_meteorology.csv")

hy <- read.csv("verification/hy_temp.csv")
hy_met <- read.csv("verification/hy_meteorology.csv")

dl <- read.csv("verification/dl_temp.csv")
dl_met <- read.csv("verification/dl_meteorology.csv")

obs_raw <- read.csv("input/observed_df_lter_hourly_wide_clean.csv") 
obs <- obs_raw[match(as.POSIXct(pb$time), as.POSIXct(obs_raw$DateTime)), ]

dl_noMod <- read.csv("verification/dlnoM_temp.csv")
dl_noMod_met <- read.csv("verification/dlnoM_meteorology.csv")

# obs <- obs_raw %>%
#   filter(DateTime >= min(as.POSIXct(pb$time)) & DateTime <= max(as.POSIXct(pb$time)))
obs <- obs[,-1]

pb_df <- physicalCalc(input = pb, 
                      area = area, 
                      depths = depths, 
                      meteo =  pb_met, 
                      hyps = hyps)

obs_df <- physicalCalc(input = obs, 
                      area = area, 
                      depths = depths, 
                      meteo =  pb_met, 
                      hyps = hyps)

hy_df <- physicalCalc(input = hy, 
                      area = area, 
                      depths = depths, 
                      meteo = hy_met, 
                      hyps = hyps)

dl_df <- physicalCalc(input = dl, 
                      area = area, 
                      depths = depths, 
                      meteo = dl_met, 
                      hyps = hyps)

dlnoM_df <- physicalCalc(input = dl_noMod, 
                      area = area, 
                      depths = depths, 
                      meteo = dl_noMod_met, 
                      hyps = hyps)


calc_rmse(pb_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(pb_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(pb_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(pb_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(pb_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_rmse(pb_df$EpiDepth, obs_df$EpiDepth)
calc_rmse(pb_df$HypoDepth, obs_df$HypoDepth)
calc_rmse(pb_df$Iso13, obs_df$Iso13)
calc_rmse(pb_df$Iso15, obs_df$Iso15)
calc_rmse(pb_df$Iso17, obs_df$Iso17)

calc_rmse(hy_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(hy_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(hy_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(hy_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(hy_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_rmse(hy_df$EpiDepth, obs_df$EpiDepth)
calc_rmse(hy_df$HypoDepth, obs_df$HypoDepth)
calc_rmse(hy_df$Iso13, obs_df$Iso13)
calc_rmse(hy_df$Iso15, obs_df$Iso15)
calc_rmse(hy_df$Iso17, obs_df$Iso17)

calc_rmse(dl_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(dl_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(dl_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(dl_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(dl_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_rmse(dl_df$EpiDepth, obs_df$EpiDepth)
calc_rmse(dl_df$HypoDepth, obs_df$HypoDepth)
calc_rmse(dl_df$Iso13, obs_df$Iso13)
calc_rmse(dl_df$Iso15, obs_df$Iso15)
calc_rmse(dl_df$Iso17, obs_df$Iso17)

calc_rmse(dlnoM_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(dlnoM_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(dlnoM_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(dlnoM_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(dlnoM_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_rmse(dlnoM_df$EpiDepth, obs_df$EpiDepth)
calc_rmse(dlnoM_df$HypoDepth, obs_df$HypoDepth)
calc_rmse(dlnoM_df$Iso13, obs_df$Iso13)
calc_rmse(dlnoM_df$Iso15, obs_df$Iso15)
calc_rmse(dlnoM_df$Iso17, obs_df$Iso17)


calc_nse(pb_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(pb_df$BottomWTR, obs_df$BottomWTR)
calc_nse(pb_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(pb_df$LakeNumber, obs_df$LakeNumber)
calc_nse(pb_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_nse(pb_df$EpiDepth, obs_df$EpiDepth)
calc_nse(pb_df$HypoDepth, obs_df$HypoDepth)
calc_nse(pb_df$Iso13, obs_df$Iso13)
calc_nse(pb_df$Iso15, obs_df$Iso15)
calc_nse(pb_df$Iso17, obs_df$Iso17)

calc_nse(hy_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(hy_df$BottomWTR, obs_df$BottomWTR)
calc_nse(hy_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(hy_df$LakeNumber, obs_df$LakeNumber)
calc_nse(hy_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_nse(hy_df$EpiDepth, obs_df$EpiDepth)
calc_nse(hy_df$HypoDepth, obs_df$HypoDepth)
calc_nse(hy_df$Iso13, obs_df$Iso13)
calc_nse(hy_df$Iso15, obs_df$Iso15)
calc_nse(hy_df$Iso17, obs_df$Iso17)

calc_nse(dl_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(dl_df$BottomWTR, obs_df$BottomWTR)
calc_nse(dl_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(dl_df$LakeNumber, obs_df$LakeNumber)
calc_nse(dl_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_nse(dl_df$EpiDepth, obs_df$EpiDepth)
calc_nse(dl_df$HypoDepth, obs_df$HypoDepth)
calc_nse(dl_df$Iso13, obs_df$Iso13)
calc_nse(dl_df$Iso15, obs_df$Iso15)
calc_nse(dl_df$Iso17, obs_df$Iso17)

calc_nse(dlnoM_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(dlnoM_df$BottomWTR, obs_df$BottomWTR)
calc_nse(dlnoM_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(dlnoM_df$LakeNumber, obs_df$LakeNumber)
calc_nse(dlnoM_df$thermoclineDepth, obs_df$thermoclineDepth)
calc_nse(dlnoM_df$EpiDepth, obs_df$EpiDepth)
calc_nse(dlnoM_df$HypoDepth, obs_df$HypoDepth)
calc_nse(dlnoM_df$Iso13, obs_df$Iso13)
calc_nse(dlnoM_df$Iso15, obs_df$Iso15)
calc_nse(dlnoM_df$Iso17, obs_df$Iso17)


calc_rmse(pb_df$N2, obs_df$N2)
calc_nse(pb_df$N2, obs_df$N2)
calc_rmse(hy_df$N2, obs_df$N2)
calc_nse(hy_df$N2, obs_df$N2)
calc_rmse(dl_df$N2, obs_df$N2)
calc_nse(dl_df$N2, obs_df$N2)
calc_rmse(dlnoM_df$N2, obs_df$N2)
calc_nse(dlnoM_df$N2, obs_df$N2)

# The palette with black:
brewer.pal(n = 8, name = 'Set2')
display.brewer.pal(n = 8, name = 'Set2')
cbp2 <- c("#66C2A5","#4DAF4A",  "#FF7F00",'black',  "#377EB8" )

linesize = 0.7
alphasize =0.95

pb_df <- pb_df %>%
  mutate(month = month(time),
         year = year(time))
obs_df <- obs_df %>%
  mutate(month = month(time),
         year = year(time))
dl_df <- dl_df %>%
  mutate(month = month(time),
         year = year(time))
hy_df <- hy_df %>%
  mutate(month = month(time),
         year = year(time))
dlnoM_df <- dlnoM_df %>%
  mutate(month = month(time),
         year = year(time))

Density_timeSeries_Hybrid <- ggplot() +
  geom_line(data = hy_df, aes(time, HypoDense, col = 'Hypolimnion'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = hy_df, aes(time, MetaDense, col = 'Metalimnion'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = hy_df, aes(time, EpiDense, col = 'Epilimnion'), linewidth = linesize, alpha = alphasize) +
  xlab('') + ylab("Water layer density (kg/m3)") +
  scale_colour_manual(values=cbp2) +
  ggtitle('Hybrid framework') +
  theme_bw() +
  theme(legend.title = element_blank()) 

Density_timeSeries_Hybrid_DL <- ggplot() +
  geom_line(data = dl_df, aes(time, HypoDense, col = 'Hypolimnion'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dl_df, aes(time, MetaDense, col = 'Metalimnion'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dl_df, aes(time, EpiDense, col = 'Epilimnion'), linewidth = linesize, alpha = alphasize) +
  xlab('') + ylab("Water layer density (-)") +
  scale_colour_manual(values=cbp2) +
  ggtitle('Deep learning model') +
  theme_bw() +
  theme(legend.title = element_blank()) 

Density_timeSeries_Hybrid <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  # geom_line(data = hy_df, aes(time, MetaDense/HypoDense), linewidth = linesize, alpha = alphasize, linetype = 'dashed') +
  xlab('') +  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'blue')) +
  ggtitle('Hybrid framework') +
  ylim(0.998, 1.0001) +
  theme_bw() +
  theme(legend.title = element_blank()) 

Density_timeSeries_Hybrid_DL <- ggplot() +
  geom_line(data = dl_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  # geom_line(data = dl_df, aes(time, MetaDense/HypoDense), linewidth = linesize, alpha = alphasize, linetype = 'dashed') +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'blue')) +
  ggtitle('Deep learning model no process') +
  ylim(0.998, 1.0001) +
  theme_bw() +
  theme(legend.title = element_blank()) 

Density_timeSeries_Hybrid_DLnoM <- ggplot() +
  geom_line(data = dlnoM_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  # geom_line(data = dl_df, aes(time, MetaDense/HypoDense), linewidth = linesize, alpha = alphasize, linetype = 'dashed') +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'blue')) +
  ggtitle('Deep learning model no module') +
  ylim(0.998, 1.0001) +
  theme_bw() +
  theme(legend.title = element_blank()) 

p3 <- Density_timeSeries_Hybrid / Density_timeSeries_Hybrid_DL / Density_timeSeries_Hybrid_DLnoM & plot_layout(guides = 'collect') &theme(legend.position = 'bottom')
ggsave(plot = p3, filename = "figs/Fig5.png", dpi = 300, width = 9, height =5, units = 'in')

Surftemp_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, SurfaceWTR, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dlnoM_df, aes(time, SurfaceWTR, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = pb_df, aes(time, SurfaceWTR, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = obs_df, aes(time, SurfaceWTR, col = 'Obs'), linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = hy_df, aes(time, SurfaceWTR, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  xlab('') + #ylab("Surface Water temperature (\u00B0C)") +
  ylab(expression(atop("Surface Water", paste(" temperature (\u00B0C)")))) +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

Bottomtemp_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, BottomWTR, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dlnoM_df, aes(time, BottomWTR, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = pb_df, aes(time, BottomWTR, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = obs_df, aes(time, BottomWTR, col = 'Obs'),  linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = hy_df, aes(time, BottomWTR, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  xlab('') +# ylab("Bottom Water temperature (\u00B0C)") +
  ylab(expression(atop("Bottom Water", paste(" temperature (\u00B0C)")))) +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 


Schmidt_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, SchmidtStability, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dlnoM_df, aes(time, SchmidtStability, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = pb_df, aes(time, SchmidtStability, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = obs_df, aes(time, SchmidtStability, col = 'Obs'),  linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = hy_df, aes(time, SchmidtStability, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  # xlab('') + ylab("Schmidt Stability (J m-2)") + ylab(bquote('Y-axis '(number^2)))
  labs(y = expression(paste("Schmidt stability (J ",m^-2,")")), x = "") +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

N2_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, N2, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dlnoM_df, aes(time, N2, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = pb_df, aes(time, N2, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = obs_df, aes(time, N2, col = 'Obs'),  linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = hy_df, aes(time, N2, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  labs(y = expression(paste("Max. bouyancy frequency (",s^-2,")")), x = "") + ylim(0,0.03)+
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

LN_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, LakeNumber, col = 'DL'), linewidth = linesize, alpha = alphasize)+
  geom_line(data = pb_df, aes(time, LakeNumber, col = 'PB'), linewidth = linesize, alpha = alphasize)+
  geom_line(data = obs_df, aes(time, LakeNumber, col = 'Obs'),  linewidth = 1.5, alpha = alphasize, linetype = 'dotdash')+
  geom_line(data = hy_df, aes(time, LakeNumber, col = 'Hybrid'), linewidth = linesize, alpha = alphasize)+
  xlab('') + ylab("Lake Number (-)") +
  scale_colour_manual(values=cbp2) +
  ylim(0,100) +
  theme_bw() +
  theme(legend.title = element_blank()) 

volumes_timeSeries <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9& year == 2018), aes(time, thermoclineDepth, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9& year == 2018), aes(time, thermoclineDepth, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2018), aes(time, thermoclineDepth, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9& year == 2018), aes(time, thermoclineDepth, col = 'Obs'), linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9& year == 2018), aes(time, thermoclineDepth, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  scale_y_reverse() +
  xlab('') + ylab("Thermocline depth (m)") +
  ggtitle("2018")+
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

Metavolumes_timeSeries <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9 & year == 2018), aes(time, HypoDepth, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9 & year == 2018), aes(time, HypoDepth, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2018), aes(time, HypoDepth, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2018), aes(time, HypoDepth, col = 'Obs'), linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9 & year == 2018), aes(time, HypoDepth, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  scale_y_reverse() +
  xlab('') + ylab("Lower metalimnion depth (m)") +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

isotherms_timeSeries <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'Obs'), linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  # geom_line(data = subset(dl_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'DL'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  scale_y_reverse() +
  xlab('') + ylab("15 \u00B0C isotherm (m)") +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 


volumes_timeSeries2 <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9& year == 2019), aes(time, thermoclineDepth, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9& year == 2019), aes(time, thermoclineDepth, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2019), aes(time, thermoclineDepth, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9& year == 2019), aes(time, thermoclineDepth, col = 'Obs'),  linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9& year == 2019), aes(time, thermoclineDepth, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  # geom_line(data = subset(dl_df, month >= 5 & month <= 9& year == 2019), aes(time, thermoclineDepth, col = 'DL'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  scale_y_reverse() +
  xlab('') + ylab("") +
  scale_colour_manual(values=cbp2) +
  ggtitle("2019")+
  theme_bw() +
  theme(legend.title = element_blank()) 

Metavolumes_timeSeries2 <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9 & year == 2019), aes(time, HypoDepth, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9 & year == 2019), aes(time, HypoDepth, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2019), aes(time, HypoDepth, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2019), aes(time, HypoDepth, col = 'Obs'), linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9 & year == 2019), aes(time, HypoDepth, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  scale_y_reverse() +
  xlab('') + ylab("") +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

isotherms_timeSeries2 <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9 & year == 2019), aes(time, Iso15, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9 & year == 2019), aes(time, Iso15, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2019), aes(time, Iso15, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2019), aes(time, Iso15, col = 'Obs'),  linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9 & year == 2019), aes(time, Iso15, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  scale_y_reverse() +
  xlab('') + ylab("") +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

LN_timeSeries2 <- ggplot() +
  geom_line(data = subset(dl_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(dlnoM_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) + # thermocline depth
  geom_line(data = subset(pb_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'Obs'), linewidth = 1.5, alpha = alphasize, linetype = 'dotdash') +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  scale_y_reverse() +
  ylim(0,10)+
  xlab('') + ylab("") +
  scale_colour_manual(values=cbp2) +
  
  theme_bw() +
  theme(legend.title = element_blank()) 

p1 <- (Surftemp_timeSeries / Bottomtemp_timeSeries /Schmidt_timeSeries /N2_timeSeries)  + plot_layout(guides = 'collect') &theme(legend.position = 'bottom')
ggsave(plot = p1, filename = "figs/Fig3.png", dpi = 300, width = 15, height =11, units = 'in')
  
p2 <-  ((volumes_timeSeries/  Metavolumes_timeSeries/
 isotherms_timeSeries ) | (volumes_timeSeries2/  Metavolumes_timeSeries2/
                             isotherms_timeSeries2 ))+ plot_layout(guides = 'collect')& theme(legend.position = 'bottom')
ggsave(plot = p2, filename = "figs/Fig4.png", dpi = 300, width = 15, height = 9, units = 'in')

p3 <- p1 / p2 + plot_layout(widths = c(2,2,2,2,2), heights = unit(c(4,4,4,4, 14), c('cm', 'cm', "cm", "cm", "cm")))
ggsave(plot = p3, filename = "figs/combined_Fig3+4.png", dpi = 300, width = 10, height = 16, units = 'in')



stn_hy <- hy_df %>%
  mutate(doy = yday(time)) %>%
  mutate(week = lubridate::isoweek(time)) %>%
  group_by(year, month) %>%
  select(time, SurfaceWTR, BottomWTR, SchmidtStability) %>%
  summarise_all(list(mean, sd))

stn_dl <- dl_df %>%
  mutate(doy = yday(time)) %>%
  mutate(week = lubridate::isoweek(time)) %>%
  group_by(year, month) %>%
  select(time,SurfaceWTR, BottomWTR, SchmidtStability) %>%
  summarise_all(list(mean, sd))

stn_dlnoM <- dlnoM_df %>%
  mutate(doy = yday(time)) %>%
  mutate(week = lubridate::isoweek(time)) %>%
  group_by(year, month) %>%
  select(time,SurfaceWTR, BottomWTR, SchmidtStability) %>%
  summarise_all(list(mean, sd))

g1 <- ggplot() +
  geom_line(data = stn_hy, aes(time_fn1            , SurfaceWTR_fn1 / SurfaceWTR_fn2, col = 'Hybrid')) +
  geom_line(data = stn_dl, aes(time_fn1            , SurfaceWTR_fn1 / SurfaceWTR_fn2, col = 'DL no prcs')) +
  geom_line(data = stn_dlnoM, aes(time_fn1            , SurfaceWTR_fn1 / SurfaceWTR_fn2, col = 'DL no mod')) +
  xlab('') + 
  ylab(expression(atop("Surface Water temperature", paste("Signal-to-Noise Ratio (-)")))) +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 
g2 <- ggplot() +
  geom_line(data = stn_hy, aes(time_fn1            , BottomWTR_fn1  / BottomWTR_fn2, col = 'Hybrid')) +
  geom_line(data = stn_dl, aes(time_fn1            , BottomWTR_fn1  / BottomWTR_fn2, col = 'DL no prcs')) +
  geom_line(data = stn_dlnoM, aes(time_fn1            , BottomWTR_fn1  / BottomWTR_fn2, col = 'DL no mod')) +
  xlab('') + 
  ylab(expression(atop("Bottom Water temperature", paste("Signal-to-Noise Ratio (-)")))) +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 
g3 <- ggplot() +
  geom_line(data = stn_hy, aes(time_fn1            , SchmidtStability_fn1  / SchmidtStability_fn2, col = 'Hybrid')) +
  geom_line(data = stn_dl, aes(time_fn1            , SchmidtStability_fn1  / SchmidtStability_fn2, col = 'DL no prcs')) +
  geom_line(data = stn_dlnoM, aes(time_fn1            , SchmidtStability_fn1  / SchmidtStability_fn2, col = 'DL no mod')) +
  xlab('') + 
  ylab(expression(atop("Schmidt stability", paste("Signal-to-Noise Ratio (-)")))) +
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 
g4 <- (g1 / g2 /g3) + plot_layout(guides = 'collect') &theme(legend.position = 'bottom')
ggsave(plot = g4, filename = "figs/Fig6.png", dpi = 300, width = 5, height =8, units = 'in')

