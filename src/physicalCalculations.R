#setwd("C:/Users/ladwi/Documents/Projects/R/LakePIAB")
 setwd("/Users/robertladwig/Documents/DSI/LakePIAB")
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

calc_rmse(pb_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(pb_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(pb_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(pb_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(pb_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_rmse(pb_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_rmse(pb_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_rmse(pb_df$Iso13, obs_df$Iso13)
calc_rmse(pb_df$Iso15, obs_df$Iso15)
calc_rmse(pb_df$Iso17, obs_df$Iso17)

calc_rmse(hy_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(hy_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(hy_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(hy_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(hy_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_rmse(hy_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_rmse(hy_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_rmse(hy_df$Iso13, obs_df$Iso13)
calc_rmse(hy_df$Iso15, obs_df$Iso15)
calc_rmse(hy_df$Iso17, obs_df$Iso17)

calc_rmse(dl_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(dl_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(dl_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(dl_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(dl_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_rmse(dl_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_rmse(dl_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_rmse(dl_df$Iso13, obs_df$Iso13)
calc_rmse(dl_df$Iso15, obs_df$Iso15)
calc_rmse(dl_df$Iso17, obs_df$Iso17)

calc_rmse(dlnoM_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_rmse(dlnoM_df$BottomWTR, obs_df$BottomWTR)
calc_rmse(dlnoM_df$SchmidtStability, obs_df$SchmidtStability)
calc_rmse(dlnoM_df$LakeNumber, obs_df$LakeNumber)
calc_rmse(dlnoM_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_rmse(dlnoM_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_rmse(dlnoM_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_rmse(dlnoM_df$Iso13, obs_df$Iso13)
calc_rmse(dlnoM_df$Iso15, obs_df$Iso15)
calc_rmse(dlnoM_df$Iso17, obs_df$Iso17)


calc_nse(pb_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(pb_df$BottomWTR, obs_df$BottomWTR)
calc_nse(pb_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(pb_df$LakeNumber, obs_df$LakeNumber)
calc_nse(pb_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_nse(pb_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_nse(pb_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
          obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_nse(pb_df$Iso13, obs_df$Iso13)
calc_nse(pb_df$Iso15, obs_df$Iso15)
calc_nse(pb_df$Iso17, obs_df$Iso17)

calc_nse(hy_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(hy_df$BottomWTR, obs_df$BottomWTR)
calc_nse(hy_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(hy_df$LakeNumber, obs_df$LakeNumber)
calc_nse(hy_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_nse(hy_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_nse(hy_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_nse(hy_df$Iso13, obs_df$Iso13)
calc_nse(hy_df$Iso15, obs_df$Iso15)
calc_nse(hy_df$Iso17, obs_df$Iso17)

calc_nse(dl_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(dl_df$BottomWTR, obs_df$BottomWTR)
calc_nse(dl_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(dl_df$LakeNumber, obs_df$LakeNumber)
calc_nse(dl_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_nse(dl_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_nse(dl_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
calc_nse(dl_df$Iso13, obs_df$Iso13)
calc_nse(dl_df$Iso15, obs_df$Iso15)
calc_nse(dl_df$Iso17, obs_df$Iso17)

calc_nse(dlnoM_df$SurfaceWTR, obs_df$SurfaceWTR)
calc_nse(dlnoM_df$BottomWTR, obs_df$BottomWTR)
calc_nse(dlnoM_df$SchmidtStability, obs_df$SchmidtStability)
calc_nse(dlnoM_df$LakeNumber, obs_df$LakeNumber)
calc_nse(dlnoM_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(thermoclineDepth))
calc_nse(dlnoM_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(EpiDepth))
calc_nse(dlnoM_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth), 
         obs_df %>% filter(month >= 6 & month <=9) %>% select(HypoDepth))
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

library(MetBrewer)
colors = met.brewer(name="Egypt", n=4, type="discrete")
cbp2 <- c(colors[2:3], colors[1], "black", colors[4])
cbp_noobs <- c(colors[2:3], colors[1],    colors[4])
linesize = 0.7
alphasize =0.95



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

hy_df <- hy_df %>%
  mutate(DensityViolation = ifelse(EpiDense/MetaDense > 1, 1, 0))

dl_df <- dl_df %>%
  mutate(DensityViolation = ifelse(EpiDense/MetaDense > 1, 1, 0))
dlnoM_df <- dlnoM_df %>%
  mutate(DensityViolation = ifelse(EpiDense/MetaDense > 1, 1, 0))

range(hy_df$EpiDense/hy_df$MetaDense, na.rm = T)
range(dl_df$EpiDense/dl_df$MetaDense, na.rm = T)
range(dlnoM_df$EpiDense/dlnoM_df$MetaDense, na.rm = T)

Density_timeSeries_Hybrid_alt <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense, linetype = 'EpiDense'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = hy_df, aes(time, MetaDense, linetype = 'MetaDense'), linewidth = linesize, alpha = alphasize) +
  geom_point(data = subset(hy_df, DensityViolation ==1), aes(time, MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black','black', 'red')) +
  ggtitle('Hybrid framework') +
  ylim(996, 1000) +
  theme_bw() +
  theme(legend.title = element_blank()) 

Density_timeSeries_Hybrid_DL_alt <- ggplot() +
  geom_line(data = dl_df, aes(time, EpiDense, linetype = 'EpiDense'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dl_df, aes(time, MetaDense, linetype = 'MetaDense'), linewidth = linesize, alpha = alphasize) +
  geom_point(data = subset(dl_df, DensityViolation ==1), aes(time, MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black','black', 'red')) +
  ggtitle('Deep learning model (no process)') +
  ylim(996, 1000) +
  theme_bw() +
  theme(legend.title = element_blank()) 

Density_timeSeries_Hybrid_DLnoM_alt <- ggplot() +
  geom_line(data = dlnoM_df, aes(time, EpiDense, linetype = 'EpiDense'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dlnoM_df, aes(time, MetaDense, linetype = 'MetaDense'), linewidth = linesize, alpha = alphasize) +
  geom_point(data = subset(dlnoM_df, DensityViolation ==1), aes(time, MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black','black', 'red')) +
  ggtitle('Pretrained Deep learning model (no module)') +
  ylim(996, 1000) +
  theme_bw() +
  theme(legend.title = element_blank()) 


p9 <- (Density_timeSeries_Hybrid_alt / Density_timeSeries_Hybrid_DL_alt / Density_timeSeries_Hybrid_DLnoM_alt) & plot_layout(guides = 'collect') & plot_annotation(tag_levels = 'A')# &theme(legend.position = 'bottom')
ggsave(plot = p9, filename = "figs/Fig5_alt.png", dpi = 300, width = 9, height =8, units = 'in')



Density_timeSeries_Hybrid <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  geom_point(data = subset(hy_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'red')) +
  ggtitle('Hybrid framework') +
  ylim(0.998, 1.0003) +
  theme_bw() +
  theme(legend.title = element_blank(), legend.position = "none") 

Density_timeSeries_Hybrid_DL <- ggplot() +
  geom_line(data = dl_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  geom_point(data = subset(dl_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'red')) +
  ggtitle('Deep learning model (no process)') +
  ylim(0.998, 1.0003) +
  theme_bw() +
  theme(legend.title = element_blank(), legend.position = "none") 

Density_timeSeries_Hybrid_DLnoM <- ggplot() +
  geom_line(data = dlnoM_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  geom_point(data = subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'red')) +
  ggtitle('Pretrained Deep learning model (no module)') +
  ylim(0.998, 1.0003) +
  theme_bw() +
  theme(legend.title = element_blank(),legend.position = "none") 

Density_timeSeries_Hybrid <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = 0) +
  geom_point(data = subset(hy_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  # geom_point(data = subset(hy_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'red')) +
  ggtitle('Hybrid framework') +
  ylim(0.9999, 1.0002) +
  theme_bw() +
  theme(legend.title = element_blank(), legend.position = "none") 

Density_timeSeries_Hybrid_DL <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = 0) +
  geom_point(data = subset(dl_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  # geom_point(data = subset(dl_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'red')) +
  ggtitle('Deep learning model (no process)') +
  ylim(0.9999, 1.0002) +
  theme_bw() +
  theme(legend.title = element_blank(), legend.position = "none") 

Density_timeSeries_Hybrid_DLnoM <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = 0) +
  geom_point(data =  subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = alphasize) +
  # geom_point(data = subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  scale_colour_manual(values=c('black', 'red')) +
  ggtitle('Pretrained Deep learning model (no module)') +
  ylim(0.9999, 1.0002) +
  theme_bw() +
  theme(legend.title = element_blank(),legend.position = "none") 

Density_timeSeries_Hybrid_all <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense/MetaDense), linewidth = linesize, alpha = 0) +
  geom_point(data = subset(hy_df, DensityViolation ==1), aes(time, EpiDense/MetaDense, col = 'Hybrid'), alpha = alphasize) +
  geom_point(data = subset(dl_df, DensityViolation ==1), aes(time, EpiDense/MetaDense, col = 'DL no prcs'),  alpha = alphasize) +
  geom_point(data =  subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense/MetaDense, col = 'DL no mod'),  alpha = alphasize) +
  # geom_point(data = subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  #ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  ylab(paste("Epilimnion:metalimnion density (-)")) +
  scale_colour_manual(values=c('black', 'red')) +
  # ggtitle('Pretrained Deep learning model (no module)') +
  scale_colour_manual(values=cbp2) +
  # scale_y_continuous(trans='log10') +
  # scale_y_continuous(labels = scientific) +
  # scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
  #               labels = trans_format("log10", math_format(10^.x))) +
  ylim(1.0, 1.0002) +
  theme_bw() +
  theme(legend.title = element_blank(),legend.position = "bottom") 

Density_timeSeries_Hybrid_all <- ggplot() +
  geom_line(data = hy_df, aes(time, EpiDense-MetaDense), linewidth = linesize, alpha = 0) +
  geom_point(data = subset(hy_df, DensityViolation ==1), aes(time, EpiDense-MetaDense, col = 'Hybrid'), alpha = alphasize) +
  geom_point(data = subset(dl_df, DensityViolation ==1), aes(time, EpiDense-MetaDense, col = 'DL no prcs'),  alpha = alphasize) +
  geom_point(data =  subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense-MetaDense, col = 'DL no mod'),  alpha = alphasize) +
  # geom_point(data = subset(dlnoM_df, DensityViolation ==1), aes(time, EpiDense/MetaDense), color = 'red', alpha = alphasize, size =0.15) +
  xlab('') +# ylab(paste("Epilimnion by",\n," metalimnion density (-)")) +
  # ylab(expression(atop("Epilimnion by", paste(" metalimnion density (-)")))) +
  # ylab(paste("Epilimnion - metalimnion density (-)")) +
  labs(y = expression(paste("Avg. epilimnion - metalimnion density (kg ",m^-3,")")), x = '') +
  scale_colour_manual(values=c('black', 'red')) +
  # ggtitle('Pretrained Deep learning model (no module)') +
  scale_colour_manual(values=cbp2) +
  # geom_hline(yintercept = 1e-4) +
  # scale_y_continuous(trans='log10') +
  # scale_y_continuous(labels = scientific) +
  # scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
  #               labels = trans_format("log10", math_format(10^.x))) +
  ylim(0, 0.2) +
  theme_bw() +
  theme(legend.title = element_blank(),legend.position = "bottom") 

p3 <- Density_timeSeries_Hybrid / Density_timeSeries_Hybrid_DL / Density_timeSeries_Hybrid_DLnoM & plot_layout(guides = 'collect') & plot_annotation(tag_levels = 'A')# &theme(legend.position = 'bottom')
ggsave(plot = Density_timeSeries_Hybrid_all, filename = "figs/Fig5.png", dpi = 300, width = 9, height =4, units = 'in')

Surftemp_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, SurfaceWTR, col = 'DL no prcs'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = dlnoM_df, aes(time, SurfaceWTR, col = 'DL no mod'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = pb_df, aes(time, SurfaceWTR, col = 'PB'), linewidth = linesize, alpha = alphasize) +
  geom_line(data = obs_df, aes(time, SurfaceWTR, col = 'Obs'), linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = obs_df, aes(time, BottomWTR, col = 'Obs'),  linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = obs_df, aes(time, SchmidtStability, col = 'Obs'),  linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = obs_df, aes(time, N2, col = 'Obs'),  linewidth = 1.5, alpha = alphasize) +
  geom_line(data = hy_df, aes(time, N2, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  labs(y = expression(paste("Max. bouyancy frequency (",s^-2,")")), x = "") + ylim(0,0.03)+
  scale_colour_manual(values=cbp2) +
  theme_bw() +
  theme(legend.title = element_blank()) 

LN_timeSeries <- ggplot() +
  geom_line(data = dl_df, aes(time, LakeNumber, col = 'DL'), linewidth = linesize, alpha = alphasize)+
  geom_line(data = pb_df, aes(time, LakeNumber, col = 'PB'), linewidth = linesize, alpha = alphasize)+
  geom_line(data = obs_df, aes(time, LakeNumber, col = 'Obs'),  linewidth = 1.5, alpha = alphasize)+
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9& year == 2018), aes(time, thermoclineDepth, col = 'Obs'), linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2018), aes(time, HypoDepth, col = 'Obs'), linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2018), aes(time, Iso15, col = 'Obs'), linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9& year == 2019), aes(time, thermoclineDepth, col = 'Obs'),  linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2019), aes(time, HypoDepth, col = 'Obs'), linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2019), aes(time, Iso15, col = 'Obs'),  linewidth = 1.5, alpha = alphasize) +
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
  geom_line(data = subset(obs_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'Obs'), linewidth = 1.5, alpha = alphasize) +
  geom_line(data = subset(hy_df, month >= 5 & month <= 9 & year == 2019), aes(time, LakeNumber, col = 'Hybrid'), linewidth = linesize, alpha = alphasize) +
  scale_y_reverse() +
  ylim(0,10)+
  xlab('') + ylab("") +
  scale_colour_manual(values=cbp2) +
  
  theme_bw() +
  theme(legend.title = element_blank()) 

p1 <- (Surftemp_timeSeries / Bottomtemp_timeSeries /Schmidt_timeSeries /N2_timeSeries)  + plot_layout(guides = 'collect') &theme(legend.position = 'bottom')
ggsave(plot = p1, filename = "figs/Fig3.png", dpi = 300, width = 15, height =11, units = 'in')

ggsave(plot = (isotherms_timeSeries | isotherms_timeSeries2), filename = "figs/IAGLR_iso15.png", dpi = 300, width = 15, height =6, units = 'in')

p2 <-  ((volumes_timeSeries/  Metavolumes_timeSeries/
 isotherms_timeSeries ) | (volumes_timeSeries2/  Metavolumes_timeSeries2/
                             isotherms_timeSeries2 ))+ plot_layout(guides = 'collect')& theme(legend.position = 'bottom')
# ggsave(plot = p2, filename = "figs/Fig4.png", dpi = 300, width = 15, height = 9, units = 'in')

p4 <- p1 / p2 + plot_layout(widths = c(2,2,2,2,2), heights = unit(c(4,4,4,4, 14), c('cm', 'cm', "cm", "cm", "cm"))) & plot_annotation(tag_levels = 'A')
ggsave(plot = p4, filename = "figs/Fig4.png", dpi = 300, width = 10, height = 16, units = 'in')



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
g4 <- (g1 / g2 /g3) + plot_layout(guides = 'collect') &theme(legend.position = 'bottom') &  plot_annotation(tag_levels = 'A')
ggsave(plot = g4, filename = "figs/Fig6.png", dpi = 300, width = 5, height =7, units = 'in')



fit_process <- data.frame(
  'variable' = c('total temp', 'surf temp', 'bot temp', 'schmidt', 'buoyancy', 'thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso'),
  'rmse' = c(4.63, 4.34, 4.02, 139.92, 0.006, 5.63, 6.93, 3.28, 4.40, 4.34, 4.16),
  'nse' = c(0.53, .77, -1.35, .79,-0.14, -4.64, -5.21, -2, -0.76, -1.09, -1.94),
  'id' = 'PB'
)
fit_hybrid <- data.frame(
  'variable' = c('total temp', 'surf temp', 'bot temp', 'schmidt', 'buoyancy', 'thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso'),
  'rmse' = c(2.14,2.12,1.73,85.43, 0.004, 2.71, 4.05, 1.64, 1.94, 1.68, 1.57),
  'nse' = c(.9, .94, .56, .92, .25, -.31, -1.13, .24, .65, .68, .57),
  'id' = 'Hybrid'
)
fit_dlnoprcs <- data.frame(
  'variable' = c('total temp', 'surf temp', 'bot temp', 'schmidt', 'buoyancy', 'thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso'),
  'rmse' = c(4.14,5.99,3.1, 255.54, 0.006, 3.16, 3.6, 2.88, 3.64, 2.94, 2.97),
  'nse' = c(0.62, 0.57, -0.4, 0.31, -0.19, -0.77, -0.68, -1.31, -0.2, 0.03, -0.5),
  'id' = 'DL no prcs'
)
fit_dlnomod <- data.frame(
  'variable' = c('total temp', 'surf temp', 'bot temp', 'schmidt', 'buoyancy', 'thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso'),
  'rmse' = c(2.11, 1.57, 2.10, 85.85, 0.005, 2.65, 4.71, 2.53, 3.88, 2.58, 2.34),
  'nse' = c(0.9, .97, .35, .92, -0.02, -0.25, -1.88, -0.78, -0.37, .25, 0.08),
  'id' = 'DL no mod'
)


depths <- ggplot() +
  geom_point(data = fit_process %>% filter(variable %in% c('thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  geom_point(data = fit_hybrid%>% filter(variable %in% c('thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  geom_point(data = fit_dlnoprcs%>% filter(variable %in% c('thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  geom_point(data = fit_dlnomod%>% filter(variable %in% c('thermocline', 'upper meta', 'lower meta', '13 iso', '15 iso', '17 iso')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  scale_colour_manual(values=cbp_noobs) +
  # facet_wrap(~ variable, scales = 'free') +
  xlab('RMSE (m)') + ylab('NSE (-)') +
  theme_bw() +
  theme(legend.title = element_blank()) 


temps <- ggplot() +
  geom_point(data = fit_process %>% filter(variable %in% c('total temp', 'surf temp', 'bot temp')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  geom_point(data = fit_hybrid%>% filter(variable %in% c('total temp', 'surf temp', 'bot temp')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  geom_point(data = fit_dlnoprcs%>% filter(variable %in% c('total temp', 'surf temp', 'bot temp')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  geom_point(data = fit_dlnomod%>% filter(variable %in% c('total temp', 'surf temp', 'bot temp')), aes(rmse, nse, shape = variable, col = id), size = 3) +
  scale_colour_manual(values=cbp_noobs) +
  # facet_wrap(~ variable, scales = 'free') +
  xlab('RMSE (\u00B0C)') + ylab('NSE (-)') +
  theme_bw() +
  theme(legend.title = element_blank()) 
g20 <- (temps | depths)+ plot_layout(guides = 'collect') &theme(legend.position = 'bottom') &  plot_annotation(tag_levels = 'A')
ggsave(plot = g20, filename = "figs/IAGLR_performance.png", dpi = 300, width = 11, height =5, units = 'in')
