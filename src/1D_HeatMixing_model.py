import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir("/home/robert/Projects/LakePIAB/src")
from oneD_HeatMixing_Functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel

## lake configurations
zmax = 25 # maximum lake depth
nx = 25 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
hyps_all = get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx, nx = nx)
                            
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = '../input/Mendota_2002.csv',
                    secchifile = None, 
                    windfactor = 1.0)

## here we define our initial profile
u_ini = initial_profile(initfile = '../input/observedTemp.txt', nx = nx, dx = dx,
                     depth = hyps_all[1],
                     processed_meteo = meteo_all[0])
                     
hydrodynamic_timestep = 24 * dt
total_runtime = 60 # 365 * 2#14

startingDate = meteo_all[0]['date'][0]

nTotalSteps = int(total_runtime * hydrodynamic_timestep/ dt)
temp = np.full([nx, nTotalSteps], np.nan)
avgtemp = np.full([nTotalSteps, 6], np.nan)
temp_initial = np.full([nx, nTotalSteps], np.nan)
temp_heat = np.full([nx, nTotalSteps], np.nan)
temp_diff = np.full([nx, nTotalSteps], np.nan)
temp_mix = np.full([nx, nTotalSteps], np.nan)
temp_conv = np.full([nx, nTotalSteps], np.nan)
temp_ice = np.full([nx, nTotalSteps], np.nan)
diff = np.full([nx, nTotalSteps], np.nan)
meteo = np.full([9, nTotalSteps], np.nan)
buoyancy = np.full([nx, nTotalSteps], np.nan)
td_depth = np.full([1, nTotalSteps], np.nan)
heatflux_lwsl = np.full([1, nTotalSteps], np.nan)
heatflux_sw = np.full([nx, nTotalSteps], np.nan)
icethickness = np.full([1, nTotalSteps], np.nan)
snowthickness = np.full([1, nTotalSteps], np.nan)
snowicethickness = np.full([1, nTotalSteps], np.nan)

Start = datetime.datetime.now()
if 'res' in locals() or 'res' in globals():
  del res
  
for i in range(total_runtime):
  if 'res' in locals() or 'res' in globals():
    u = res['temp'][:,-1]
    startTime = res['endtime']
    endTime = res['endtime'] + hydrodynamic_timestep - 1
    ice = res['iceflag']
    Hi = res['last_ice']
    iceT = res['icemovAvg']
    supercooled = res['supercooled']
    kd_light = 0.8
    matrix_range_start = deepcopy(matrix_range_end)# max(0, round(startTime/dt))
    matrix_range_end = matrix_range_start + 24# round(endTime/dt) 
  else:
    u = deepcopy(u_ini)
    startTime = 1
    endTime = hydrodynamic_timestep - 1
    ice = False
    Hi = 0
    iceT = 6
    supercooled = 0
    kd_light = 0.8
    matrix_range_start = 0 #max(0, round(startTime/dt))
    matrix_range_end = 24 #round(endTime/dt)
    
  res = run_thermalmodel(
    u = u,
    startTime = startTime, 
    endTime =  endTime,
    area = hyps_all[0],
    volume = hyps_all[2],
    depth = hyps_all[1],
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    daily_meteo = meteo_all[0],
    secview = meteo_all[1],
    ice = ice,
    Hi = Hi,
    iceT = iceT,
    supercooled = supercooled,
    scheme='explicit',
    kd_light = kd_light,
    denThresh=1e-3,
    albedo = 0.1,
    eps=0.97,
    emissivity=0.97,
    sigma=5.67e-8,
    sw_factor = 1.0,
    p2=1,
    B=0.61,
    g=9.81,
    Cd = 0.0013, # momentum coeff (wind)
    meltP=1,
    dt_iceon_avg=0.8,
    Hgeo=0.1, # geothermal heat
    KEice=0,
    Ice_min=0.1,
    pgdl_mode = 'on')
  
  temp[:, matrix_range_start:(matrix_range_end)] =  res['temp']
  diff[:, matrix_range_start:matrix_range_end] =  res['diff']
  avgtemp[matrix_range_start:matrix_range_end,:] = res['average'].values
  temp_initial[:, matrix_range_start:matrix_range_end] =  res['temp_initial']
  temp_heat[:, matrix_range_start:matrix_range_end] =  res['temp_heat']
  temp_diff[:, matrix_range_start:matrix_range_end] =  res['temp_diff']
  temp_mix[:, matrix_range_start:matrix_range_end] =  res['temp_mix']
  temp_conv[:, matrix_range_start:matrix_range_end] =  res['temp_conv']
  temp_ice[:, matrix_range_start:matrix_range_end] =  res['temp_ice']
  meteo[:, matrix_range_start:matrix_range_end] =  res['meteo_input']
  buoyancy[:, matrix_range_start:matrix_range_end] = res['buoyancy_pgdl']
  td_depth[0, matrix_range_start:matrix_range_end] = res['thermoclinedepth']
  heatflux_lwsl[0, matrix_range_start:matrix_range_end] = res['heatflux_lwsl']
  heatflux_sw[:, matrix_range_start:matrix_range_end] = res['heatflux_sw']
  icethickness[0, matrix_range_start:matrix_range_end] = res['icethickness']
  snowthickness[0, matrix_range_start:matrix_range_end] = res['snowthickness']
  snowicethickness[0, matrix_range_start:matrix_range_end] = res['snowicethickness']
  
# convert averages from array to data frame
avgtemp_df = pd.DataFrame(avgtemp, columns=["time", "thermoclineDep", "epi", "hypo", "tot", "stratFlag"])
avgtemp_df.insert(2, "icethickness", icethickness[0,], True)
avgtemp_df.insert(2, "snowthickness", snowthickness[0,], True)
avgtemp_df.insert(2, "snowicethickness", snowicethickness[0,], True)

End = datetime.datetime.now()
print(End - Start)

# epi/hypo/total
colors = ['#F8766D', '#00BA38', '#619CFF']
avgtemp_df.plot(x='time', y=['epi', 'hypo', 'tot'], color=colors, kind='line')
plt.show()

# stratflag
avgtemp_df.plot(x='time', y=['stratFlag'], kind='line', color="black")
plt.show()

# thermocline depth
avgtemp_df.plot(x='time', y=['thermoclineDep'], color="black")
plt.gca().invert_yaxis()
plt.scatter(avgtemp_df.time, avgtemp_df.stratFlag, c=avgtemp_df.stratFlag)
plt.show()

# ice thickness
avgtemp_df.plot(x='time', y=['icethickness'], color="black")
plt.show()

# snowice thickness
avgtemp_df.plot(x='time', y=['snowicethickness'], color="black")
plt.show()

# snow thickness
avgtemp_df.plot(x='time', y=['snowthickness'], color="black")
plt.show()

# heatmap of temps  
plt.subplots(figsize=(40,40))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of diffusivities  
plt.subplots(figsize=(40,40))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

plt.subplots(figsize=(40,40))
sns.heatmap(heatflux_sw, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

x_time = np.full([1, nTotalSteps], np.nan)
x_time[0, 0:matrix_range_end] = np.array(range(0,matrix_range_end))
plt.plot(x_time[0,:], heatflux_lwsl[0,:], color = 'black')
plt.show()

dataframe = pd.DataFrame(heatflux_lwsl) 
dataframe.to_csv(r"heatflux_python.csv")
