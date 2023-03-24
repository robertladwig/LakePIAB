setwd("C:/Users/ladwi/Documents/Projects/R/LakePIAB")
library(tidyverse)
library(rLakeAnalyzer)
library(patchwork)
library(lubridate)

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
                  "BottomWTR" = NULL)
  
  for (i in 1:nrow(pb)){
    
    print(round((100 * i) / nrow(pb)), 2)
    
    wtr <- as.numeric(temp[i,])
    wtr[wtr == -999] = NA
    
    TD <- thermo.depth(wtr = wtr,depths = depths)
    
    St <- schmidt.stability(wtr = wtr,
                            depths = depths,
                            bthA = hyps$Area_meterSquared,
                            bthD = hyps$Depth_meter)
    
    MetaDepths <- meta.depths(wtr = wtr,
                depths = depths, seasonal = T)
    
    if (any(is.na(MetaDepths))){
      EpiDense <- NA
      HypoDense <- NA
      WindVel <- NA
      LN <- NA
    } else {
      
      EpiDense <- layer.density(top = 0, bottom = MetaDepths[1], wtr = wtr,
                                depths = depths, bthA = hyps$Area_meterSquared, bthD = hyps$Depth_meter)
      
      HypoDense <- layer.density(top = MetaDepths[2], bottom = max(depths), wtr = wtr,
                                 depths = depths, bthA = hyps$Area_meterSquared, bthD = hyps$Depth_meter)
      
      WindVel <- uStar(wndSpeed = meteo$Uw[i], wndHeight = 10, averageEpiDense = EpiDense)
      
      LN <- lake.number(bthA = hyps$Area_meterSquared, 
                        bthD = hyps$Depth_meter, 
                        uStar = WindVel, 
                        St = St, 
                        metaT = MetaDepths[1], 
                        metaB = MetaDepths[2], 
                        averageHypoDense = HypoDense)
      
      SurfTemp <- wtr[2]
      
      BottomTemp <- wtr[length(wtr)-1]
    }
    
    
    
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
                    "BottomWTR" = BottomTemp))
  }
  
  return(df)
  
}

calc_rmse <- function(sim, obs){
  idx = !is.na(obs)
  rmse = sqrt(sum((sim[idx] - obs[idx])**2, na.rm = T) / length(obs[idx]))
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


temp_timeSeries <- ggplot() +
  geom_line(data = pb_df, aes(time, SurfaceWTR, col = 'Surface PB')) +
  geom_line(data = pb_df, aes(time, BottomWTR, col = 'Bottom PB')) +
  geom_line(data = obs_df, aes(time, SurfaceWTR, col = 'Surface Obs')) +
  geom_line(data = obs_df, aes(time, BottomWTR, col = 'Bottom Obs')) +
  geom_line(data = hy_df, aes(time, SurfaceWTR, col = 'Surface Hybrid')) +
  geom_line(data = hy_df, aes(time, BottomWTR, col = 'Bottom Hybrid')) +
  geom_line(data = dl_df, aes(time, SurfaceWTR, col = 'Surface DL')) +
  geom_line(data = dl_df, aes(time, BottomWTR, col = 'Bottom DL')) +
  xlab('') + ylab("Water temperature (deg C)") +
  theme_bw()

Schmidt_timeSeries <- ggplot() +
  geom_line(data = pb_df, aes(time, SchmidtStability, col = 'PB')) +
  geom_line(data = obs_df, aes(time, SchmidtStability, col = 'Obs')) +
  geom_line(data = hy_df, aes(time, SchmidtStability, col = 'Hybrid')) +
  geom_line(data = dl_df, aes(time, SchmidtStability, col = 'DL')) +
  xlab('') + ylab("Schmidt Stability (J m-2)") +
  theme_bw()

LN_timeSeries <- ggplot() +
  geom_line(data = pb_df, aes(time, LakeNumber, col = 'PB'))+
  geom_line(data = obs_df, aes(time, LakeNumber, col = 'Obs'))+
  geom_line(data = hy_df, aes(time, LakeNumber, col = 'Hybrid'))+
  geom_line(data = dl_df, aes(time, LakeNumber, col = 'DL'))+
  xlab('') + ylab("Lake Number (-)") +
  ylim(0,100) +
  theme_bw()

volumes_timeSeries <- ggplot() +
  geom_line(data = pb_df, aes(time, EpiDepth, col = 'PB upper metalimnion depth')) +
  geom_line(data = pb_df, aes(time, HypoDepth, col = 'PB lower metalimnion depth')) +
  geom_line(data = pb_df, aes(time, thermoclineDepth, col = 'PB thermocline depth')) +
  geom_line(data = obs_df, aes(time, EpiDepth, col = 'Obs upper metalimnion depth')) +
  geom_line(data = obs_df, aes(time, HypoDepth, col = 'Obs lower metalimnion depth')) +
  geom_line(data = obs_df, aes(time, thermoclineDepth, col = 'Obs thermocline depth')) +
  geom_line(data = hy_df, aes(time, EpiDepth, col = 'Hybrid upper metalimnion depth')) +
  geom_line(data = hy_df, aes(time, HypoDepth, col = 'Hybrid lower metalimnion depth')) +
  geom_line(data = hy_df, aes(time, thermoclineDepth, col = 'Hybrid thermocline depth')) +
  geom_line(data = dl_df, aes(time, EpiDepth, col = 'DL upper metalimnion depth')) +
  geom_line(data = dl_df, aes(time, HypoDepth, col = 'DL lower metalimnion depth')) +
  geom_line(data = dl_df, aes(time, thermoclineDepth, col = 'DL thermocline depth')) +
  scale_y_reverse() +
  xlab('') + ylab("Depths of density gradients (m)") +
  theme_bw()

isotherms_timeSeries <- ggplot() +
  geom_line(data = pb_df, aes(time, Iso13, col = 'PB 13 deg C isotherm')) +
  geom_line(data = pb_df, aes(time, Iso15, col = 'PB 15 deg C isotherm')) +
  geom_line(data = pb_df, aes(time, Iso17, col = 'PB 17 deg C isotherm')) +
  geom_line(data = obs_df, aes(time, Iso13, col = 'Obs 13 deg C isotherm')) +
  geom_line(data = obs_df, aes(time, Iso15, col = 'Obs 15 deg C isotherm')) +
  geom_line(data = obs_df, aes(time, Iso17, col = 'Obs 17 deg C isotherm')) +
  geom_line(data = hy_df, aes(time, Iso13, col = 'Hybrid 13 deg C isotherm')) +
  geom_line(data = hy_df, aes(time, Iso15, col = 'Hybrid 15 deg C isotherm')) +
  geom_line(data = hy_df, aes(time, Iso17, col = 'Hybrid 17 deg C isotherm')) +
  geom_line(data = dl_df, aes(time, Iso13, col = 'DL 13 deg C isotherm')) +
  geom_line(data = dl_df, aes(time, Iso15, col = 'DL 15 deg C isotherm')) +
  geom_line(data = dl_df, aes(time, Iso17, col = 'DL 17 deg C isotherm')) +
  scale_y_reverse() +
  xlab('') + ylab("Isotherms (m") +
  theme_bw()

temp_timeSeries/ isotherms_timeSeries/volumes_timeSeries/ Schmidt_timeSeries /LN_timeSeries
