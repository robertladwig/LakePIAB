import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch

os.chdir("/home/robert/Projects/LakePIAB/src")
from oneD_HeatMixing_Functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel, run_hybridmodel

## get normalization variables from deep learning
device = torch.device('cpu')

data_df = pd.read_csv("./../MCL/02_training/all_data_lake_modeling_in_time.csv")
data_df = data_df.fillna('')
time = data_df['time']
data_df = data_df.drop(columns=['time'])

m0_input_columns = ['depth', 'AirTemp_degC', 'Longwave_Wm-2', 'Latent_Wm-2', 'Sensible_Wm-2', 'Shortwave_Wm-2',
                'lightExtinct_m-1','Area_m2', 
                 'day_of_year', 'time_of_day', 'ice', 'snow', 'snowice', 'temp_initial00']
m0_input_column_ix = [data_df.columns.get_loc(column) for column in m0_input_columns]

data_df_scaler = data_df[data_df.columns[m0_input_column_ix]]

training_frac = 0.60
depth_steps = 25
number_days = len(data_df_scaler)//depth_steps
n_obs = int(number_days*training_frac)*depth_steps

data = data_df_scaler.values

train_data = data[:n_obs]
test_data = data[n_obs:]

train_time = time[:n_obs]
test_time = time[n_obs:]

#performing normalization on all the columns
scaler_input = StandardScaler()
scaler_input.fit(train_data)
train_data = scaler_input.transform(train_data)

train_mean = scaler_input.mean_
train_std = scaler_input.scale_

training_frac = 0.60
depth_steps = 25
number_days = len(data_df)//depth_steps
n_obs = int(number_days*training_frac)*depth_steps

#
data = data_df.values

train_data = data[:n_obs]
test_data = data[n_obs:]

train_time = time[:n_obs]
test_time = time[n_obs:]

#performing normalization on all the columns
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)

train_mean = scaler.mean_
train_std = scaler.scale_

m0_output_columns = ['temp_heat01']
m0_output_column_ix = [data_df.columns.get_loc(column) for column in m0_output_columns]

std_scale = torch.tensor(train_std[m0_output_column_ix[0]]).to(device).numpy()
mean_scale = torch.tensor(train_mean[m0_output_column_ix[0]]).to(device).numpy()

                  
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
                     
hydrodynamic_timestep = 24 * dt
total_runtime =  365 *3 # 14 * 365
startTime = 1#150 * 24 * 3600
endTime =  (startTime + total_runtime * hydrodynamic_timestep) - 1

startingDate = meteo_all[0]['date'][1]
endingDate = meteo_all[0]['date'][(startTime + total_runtime) * hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(startTime + total_runtime * hydrodynamic_timestep/dt) - 1]

#26280
times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime * hydrodynamic_timestep/ dt)

## here we define our initial profile
u_ini = initial_profile(initfile = '../input/observedTemp.txt', nx = nx, dx = dx,
                     depth = hyps_all[1],
                     startDate = startingDate)

# u_ini = u_ini * 0 + 0.5

Start = datetime.datetime.now()

res = run_hybridmodel(
    u = deepcopy(u_ini),
    startTime = startTime, 
    endTime =  (startTime + total_runtime * hydrodynamic_timestep) - 1,
    area = hyps_all[0],
    volume = hyps_all[2],
    depth = hyps_all[1],
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    std_scale = std_scale,
    mean_scale = mean_scale,
    scaler = scaler_input,
    daily_meteo = meteo_all[0],
    secview = meteo_all[1],
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,
    scheme='implicit',
    kd_light = 0.8,
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
    pgdl_mode = 'on',
    rho_snow = 250)


temp=  res['temp']
diff =  res['diff']
avgtemp = res['average'].values
temp_initial =  res['temp_initial']
temp_heat=  res['temp_heat']
temp_diff=  res['temp_diff']
temp_mix =  res['temp_mix']
temp_conv =  res['temp_conv']
temp_ice=  res['temp_ice']
meteo=  res['meteo_input']
buoyancy = res['buoyancy_pgdl']
td_depth= res['thermoclinedepth']
heatflux_lwsl= res['heatflux_lwsl']
heatflux_sw= res['heatflux_sw']
icethickness= res['icethickness']
snowthickness= res['snowthickness']
snowicethickness= res['snowicethickness']


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

# heatmap of diffusivities  
plt.subplots(figsize=(40,40))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()
    
# heatmap of temps  
plt.subplots(figsize=(40,40))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()