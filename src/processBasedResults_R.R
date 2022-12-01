#' Modular compositional learing project -
#' Postdoc in a Box (MCL PIAB)
#' Long-term Mendota data were obtained from North Temperate Lakes Long Term
#' Ecological Research program (#DEB-1440297)
#' @author: Robert Ladwig
#' @email: ladwigjena@gmail.com

## CLEAN WORKSPACE
rm(list = ls())

# SET WD TO CURRENT DIR
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

## INSTALL PACKAGE
# install.packages("remotes")
# require(remotes)
# remotes::install_github("robertladwig/LakeModelR")

## LOAD PACKAGE(S)
library(LakeModelR)
require(tidyverse)

## GENERAL LAKE CONFIGURATION
zmax = 25 # maximum lake depth
nx = 25 # number of layers we want to have
dt = 3600  # temporal step (here, one hour because it fits boundary data)
dx = zmax/nx # spatial step

## HYPSOGRAPHY OF THE LAKE
hyps_all <- get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx,
                            nx = nx)

## ATMOSPHERIC BOUNDARY CONDITIONS
cut_meteo <- read_csv('../input/Mendota_1980_2019_box_5_CT.csv')
years <- lubridate::year(cut_meteo$datetime)
cut_meteo <- cut_meteo[which(years >= 2006), ]
write.csv(x = cut_meteo, file = '../input/Mendota_2002.csv', row.names = FALSE)

meteo_all <- provide_meteorology(meteofile = '../input/Mendota_2002.csv',
                                 secchifile = NULL)

### TIME INFORMATION
startingDate <- meteo_all[[1]]$datetime[1]
startTime = 1
endTime = 14 * 365 *24 * 3600 # seconds
total_runtime = endTime / 24 / 3600 # days

# INTERPOLATE ATMOSPHERIC BOUNDARY CONDITIONS
meteo = get_interp_drivers(meteo_all = meteo_all,
                           total_runtime = total_runtime,
                           dt = dt,
                           method = "integrate",
                           secchi = F)

## DEFINE INITIAL WATER TEMPERATURE FROM OBSERVED DATA
u_ini <- initial_profile(initfile = system.file('extdata', 'observedTemp.txt',
                                                package = 'LakeModelR'),
                         nx = nx, dx = dx,
                         depth = hyps_all[[2]],
                         processed_meteo = meteo_all[[1]])

## RUN THE LAKE MODEL
res <-  run_thermalmodel(u = u_ini,
                         startTime = startTime,
                         endTime =  endTime,
                         ice = FALSE,
                         Hi = 0,
                         iceT = 6,
                         supercooled = 0,
                         kd_light = 0.8,
                         sw_factor = 1.0,
                         zmax = zmax,
                         nx = nx,
                         dt = dt,
                         dx = dx,
                         area = hyps_all[[1]], # area
                         depth = hyps_all[[2]], # depth
                         volume = hyps_all[[3]], # volume
                         daily_meteo = meteo,
                         Cd = 0.0013,
                         scheme = 'implicit',
                         pgdl_mode = TRUE)

## SAVE THE RESULTS
temp = res$temp
mixing = res$mixing
ice = res$icethickness
snow = res$snowthickness
snowice = res$snowicethickness
avgtemp = res$average
temp_initial = res$temp_initial
temp_heat =  res$temp_heat
temp_diff =  res$temp_diff
temp_mix =  res$temp_mix
temp_conv=  res$temp_conv
temp_ice =  res$temp_ice
meteo_output =  res$meteo_input
buoyancy = res$buoyancy_pgdl
diff = res$diff

## POST-PROCESSING OF THE RESULTS
time =  startingDate + seq(1, ncol(temp), 1) * dt
# time =  as.POSIXct('2002-01-01 00:00:00', tz = 'CDT') + seq(1, ncol(temp), 1) * dt
avgtemp = as.data.frame(avgtemp)
colnames(avgtemp) = c('time', 'epi', 'hyp', 'tot', 'stratFlag', 'thermoclineDep')
avgtemp$time = time

## CREATE DATAFRAME FOR FULL TEMPERATURE PROFILES
df <- data.frame(cbind(time, t(temp)) )
colnames(df) <- c("time", as.character(paste0(seq(1,nrow(temp)))))
m.df <- reshape2::melt(df, "time")
m.df$time <- time

## CREATE DATAFRAME FOR ICE
df.ice = data.frame('time' = time,
                    'ice_h' = ice,
                    'snow_h' = snow,
                    'snowice_h' = snowice)

## HEATMAP OF WATER TEMPERATURE WITH THERMOCLINE DEPTH AND ICE THICKNESS
g <- ggplot(m.df, aes((time), dx*as.numeric(as.character(variable)))) +
  geom_raster(aes(fill = as.numeric(value)), interpolate = TRUE) +
  scale_fill_gradientn(limits = c(-1,30),
                       colours = rev(RColorBrewer::brewer.pal(11, 'Spectral')))+
  theme_minimal()  +xlab('Time') +
  ylab('Depth [m]') +
  labs(fill = 'Temp [degC]')+
  geom_line(data = avgtemp, aes(time, thermoclineDep, col = 'thermocline depth'), linetype = 'dashed', col = 'brown') +
  geom_line(data = df.ice, aes(time, ice_h * (-1), col = 'ice thickness'), linetype = 'solid', col = 'darkblue') +
  scale_y_reverse()
ggsave(filename = '../figs/heatmap.png', plot = g, width = 10, height = 5, units = 'in', dpi = 300)

## SAVE DATA FOR PGDL
df <- data.frame(cbind(time, t(temp_initial)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_initial00_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/temp_initial00.csv', row.names = F)

df <- data.frame(cbind(time, t(temp_heat)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_heat01_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/temp_heat01.csv', row.names = F)

df <- data.frame(cbind(time, t(temp_diff)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_diff02_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/temp_diff02.csv', row.names = F)

df <- data.frame(cbind(time, t(temp_mix)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_mix03_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/temp_mix03.csv', row.names = F)

df <- data.frame(cbind(time, t(temp_conv)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_conv04_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/temp_conv04.csv', row.names = F)

df <- data.frame(cbind(time, t(temp)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_total05_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/temp_total05.csv', row.names = F)


df <- data.frame(cbind(time, t(diff)) )
colnames(df) <- c("time", as.character(paste0('diffM2s-1_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/diff.csv', row.names = F)

df <- data.frame(cbind(time, t(buoyancy)) )
colnames(df) <- c("time", as.character(paste0('n2S-2_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/buoyancy.csv', row.names = F)

df <- data.frame(cbind(time, t(meteo_output)) )
colnames(df) <- c("time", "AirTemp_degC", "Longwave_Wm-2",
                  "Latent_Wm-2", "Sensible_Wm-2", "Shortwave_Wm-2",
                  "lightExtinct_m-1","ShearVelocity_mS-1", "ShearStress_Nm-2",
                  "Area_m2")
df$time <- time
df <- df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')
write.csv(df, file = '../output/meteorology_input.csv', row.names = F)


# GET HIGH-FREQUENCY OBSERVED DATA
# Package ID: knb-lter-ntl.130.29 Cataloging System:https://pasta.edirepository.org.
# Data set title: North Temperate Lakes LTER: High Frequency Water Temperature Data - Lake  Mendota Buoy 2006 - current.
inUrl2  <- "https://pasta.lternet.edu/package/data/eml/knb-lter-ntl/130/29/63d0587cf326e83f57b054bf2ad0f7fe"
infile2 <- tempfile()
try(download.file(inUrl2,infile2,method="curl"))
if (is.na(file.size(infile2))) download.file(inUrl2,infile2,method="auto")

dt2 <-read.csv(infile2,header=F
               ,skip=1
               ,sep=","
               ,quot='"'
               , col.names=c(
                 "sampledate",
                 "year4",
                 "month",
                 "daynum",
                 "hour",
                 "depth",
                 "wtemp",
                 "flag_wtemp"    ), check.names=TRUE)

unlink(infile2)

# attempting to convert dt2$sampledate dateTime string to R date structure (date or POSIXct)
tmpDateFormat<-"%Y-%m-%d"
tmp2sampledate<-as.Date(dt2$sampledate,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(length(tmp2sampledate) == length(tmp2sampledate[!is.na(tmp2sampledate)])){dt2$sampledate <- tmp2sampledate } else {print("Date conversion failed for dt2$sampledate. Please inspect the data and do the date conversion yourself.")}
rm(tmpDateFormat,tmp2sampledate)
if (class(dt2$year4)=="factor") dt2$year4 <-as.numeric(levels(dt2$year4))[as.integer(dt2$year4) ]
if (class(dt2$year4)=="character") dt2$year4 <-as.numeric(dt2$year4)
if (class(dt2$month)=="factor") dt2$month <-as.numeric(levels(dt2$month))[as.integer(dt2$month) ]
if (class(dt2$month)=="character") dt2$month <-as.numeric(dt2$month)
if (class(dt2$daynum)=="factor") dt2$daynum <-as.numeric(levels(dt2$daynum))[as.integer(dt2$daynum) ]
if (class(dt2$daynum)=="character") dt2$daynum <-as.numeric(dt2$daynum)
if (class(dt2$depth)=="factor") dt2$depth <-as.numeric(levels(dt2$depth))[as.integer(dt2$depth) ]
if (class(dt2$depth)=="character") dt2$depth <-as.numeric(dt2$depth)
if (class(dt2$wtemp)=="factor") dt2$wtemp <-as.numeric(levels(dt2$wtemp))[as.integer(dt2$wtemp) ]
if (class(dt2$wtemp)=="character") dt2$wtemp <-as.numeric(dt2$wtemp)
if (class(dt2$flag_wtemp)!="factor") dt2$flag_wtemp<- as.factor(dt2$flag_wtemp)


dt2$bhour <- ifelse(dt2$hour %/% 100 >= 1, dt2$hour/100, dt2$hour)
dt2$datetime <- as.POSIXct(paste0(dt2$sampledate,' ',dt2$bhour,':00:00'), format = "%Y-%m-%d %H:%M:%S")

obs <- dt2 %>%
  select(datetime, depth, wtemp)

idx <- (match(as.POSIXct(obs$datetime), as.POSIXct(df$time) ))

obs <- obs[which(!is.na(idx)), ]

idz <- which(obs$depth %in% seq(0,24,1))
obs = obs[idz,]

obs <- data.frame(obs)
obs$depth <- factor(obs$depth)

wide.obs <- reshape(obs, idvar = "datetime", timevar = "depth", direction = "wide")

m.wide.obs <- reshape2::melt(wide.obs, "datetime")
m.wide.obs$time <- wide.obs$time

ggplot(m.wide.obs, aes((datetime), as.numeric(variable))) +
  geom_raster(aes(fill = as.numeric(value)), interpolate = TRUE) +
  scale_fill_gradientn(limits = c(-2,35),
                       colours = rev(RColorBrewer::brewer.pal(11, 'Spectral')))+
  theme_minimal()  +xlab('Time') +
  ylab('Depth') +
  labs(fill = 'Temp [degC]')+
  scale_y_reverse()


df <- data.frame(cbind(time, t(temp)) )
colnames(df) <- c("time", as.character(paste0('tempDegC_total05_',seq(1,nrow(temp)))))
df$time <- time
df <- df %>%
  mutate(year = lubridate::year(time), doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-c('doy', 'year'))

wide.obs$datetime = as.POSIXct(format(wide.obs$datetime), tz = 'CDT')
dat0 = wide.obs[,-1]
dat0[dat0 < 5] = NA
dat = zoo::na.approx(dat0, na.rm = F, rule = 2)
dat3 = apply(as.matrix(dat), 1, function(x) approx(seq(0,20,1),x,seq(1,25,1), method = 'linear', rule=2)$y)
dat4 = t(dat3)

# dat5 = apply(as.matrix(dat4), 2, function(x) approx(x = wide.obs$datetime - wide.obs$datetime[1],
#                                                     y = x,
#                                                     xout = df$time - df$time[1],
#                                                     method = 'linear', rule=2)$y)
# dat5 = apply(as.matrix(dat4), 2, function(x) approx(as.numeric(wide.obs$datetime - df$time[1], units = 'secs'),
#                                                     x,
#                                                     as.numeric(df$time-df$time[1], units = 'secs'),
#                                                     method = 'linear', rule=2)$y)

dat5 = apply(as.matrix(dat4), 2, function(x) approx(x = wide.obs$datetime,
                                                    y = x,
                                                    xout = df$time,
                                                    method = 'linear', rule=2)$y)

dat.df <- data.frame(cbind(df$time, dat5))
colnames(dat.df) <- c("time", as.character(paste0('tempDegC_total05_',seq(1,nrow(temp)))))
dat.df$time <- df$time
dat.df <- dat.df %>%
  mutate(doy = lubridate::yday(time)) %>%
  filter(between(doy,   lubridate::yday('2009-06-04 09:00:00'),  lubridate::yday('2009-08-01 00:00:00'))) %>%
  select(-'doy')


m.dat.df <- reshape2::melt(dat.df, "time")
m.dat.df$time <- df$time

ggplot(m.dat.df, aes((time), as.numeric(variable))) +
  geom_raster(aes(fill = as.numeric(value)), interpolate = TRUE) +
  scale_fill_gradientn(limits = c(-2,35),
                       colours = rev(RColorBrewer::brewer.pal(11, 'Spectral')))+
  theme_minimal()  +xlab('Time') +
  ylab('Depth') +
  labs(fill = 'Temp [degC]')+
  scale_y_reverse()

write.csv(dat.df, file = '../output/observed_temp.csv', row.names = F)
