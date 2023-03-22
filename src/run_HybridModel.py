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
#os.chdir("C:/Users/ladwi/Documents/Projects/R/LakePIAB/src")
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel, run_thermalmodel, heating_module, diffusion_module, mixing_module, convection_module, ice_module, run_thermalmodel_hybrid, run_thermalmodel_hybrid_v2

## get normalization variables from deep learning
device = torch.device('cpu')

data_df = pd.read_csv("./../MCL/02_training/all_data_lake_modeling_in_time.csv")
data_df = data_df.fillna('')
time = data_df['time']
data_df = data_df.drop(columns=['time'])

#m0_input_columns = ['depth', 'AirTemp_degC', 'Longwave_Wm-2', 'Latent_Wm-2', 'Sensible_Wm-2', 'Shortwave_Wm-2',
 #               'lightExtinct_m-1','Area_m2', 
  #               'day_of_year', 'time_of_day', 'ice', 'snow', 'snowice', 'temp_initial00']
m0_input_columns = ['depth', 'Area_m2', 'Uw',
                 'buoyancy', 'day_of_year', 'time_of_day',  'ice', 'snow', 'snowice','diffusivity', 'temp_initial00', 'temp_heat01', 'temp_total05']
m0_input_column_ix = [data_df.columns.get_loc(column) for column in m0_input_columns]

data_df_scaler = data_df[data_df.columns[m0_input_column_ix]]

training_frac = 0.60
depth_steps = 50
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

# scaling for target
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

m0_output_columns = ['temp_diff02']
m0_output_column_ix = [data_df.columns.get_loc(column) for column in m0_output_columns]

std_scale = torch.tensor(train_std[m0_output_column_ix[0]]).to(device).numpy()
mean_scale = torch.tensor(train_mean[m0_output_column_ix[0]]).to(device).numpy()

std_input = torch.tensor(train_std[m0_input_column_ix]).to(device).numpy()
mean_input = torch.tensor(train_mean[m0_input_column_ix]).to(device).numpy()

                  
## lake configurations
zmax = 25 # maximum lake depth
nx = 25 * 2# number of layers we will have
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
total_runtime =  365 * hydrodynamic_timestep/dt  #365 *1 # 14 * 365
startTime =   (0 + 365*13) * hydrodynamic_timestep/dt #150 * 24 * 3600
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime - 1)]#[(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]
# endingDate = meteo_all[0]['date'][(startTime + total_runtime * hydrodynamic_timestep/dt) - 1]


#26280
times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime) #  * hydrodynamic_timestep/ dt)

## here we define our initial profile
u_ini = initial_profile(initfile = '../input/observedTemp.txt', nx = nx, dx = dx,
                     depth = hyps_all[1],
                     startDate = startingDate)

Start = datetime.datetime.now()

res = run_thermalmodel_hybrid(
    u = deepcopy(u_ini),
    startTime = startTime, 
    endTime =  endTime, #(startTime + total_runtime * hydrodynamic_timestep) - 1,
    area = hyps_all[0][:-1],
    volume = hyps_all[2][:-1],
    depth = hyps_all[1][:-1],
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    std_scale = std_scale,
    mean_scale = mean_scale,
    std_input = std_input,
    mean_input = mean_input,
    scaler = scaler_input,
    test_input = data_df_scaler.head(n=50),
    daily_meteo = meteo_all[0],
    secview = meteo_all[1],
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,#    
    diffusion_method = 'hendersonSellers',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme='implicit',
    km = 4 * 10**(-6), 
    weight_kz = 0.5,
    kd_light = 0.8,
    denThresh=1e-2,
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
buoyancy = res['buoyancy']
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
N_pts = 6

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 35)
ax.set_ylabel("Depth", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Hybrid Temperature")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
plt.show()

dt = pd.read_csv('../input/observed_df_lter_hourly_wide.csv', index_col=0)
dt=dt.rename(columns = {'DateTime':'time'})
dt['time'] = pd.to_datetime(dt['time'], format='%Y-%m-%d %H')
dt_red = dt[dt['time'] >= startingDate]
dt_red = dt_red[dt_red['time'] <= endingDate]
dt_notime = dt_red.drop(dt_red.columns[[0]], axis = 1)
dt_notime = dt_notime.transpose()
dt_obs = dt_notime.to_numpy()
dt_obs.shape
temp.shape

number_days =temp.shape[1]
training_frac = 0.6
n_obs = int(number_days*training_frac)

rmse = sqrt(sum(sum((temp - dt_obs)**2)) / (temp.shape[0] * temp.shape[1]))
train = sqrt(sum(sum((temp[:,0:n_obs] - dt_obs[:,0:n_obs])**2)) / (temp.shape[0] * n_obs))
test = sqrt(sum(sum((temp[:,(n_obs+1):temp.shape[1]] - dt_obs[:,(n_obs+1):temp.shape[1]])**2)) / (temp.shape[0] * (temp.shape[1] - n_obs)))

sqrt(sum((temp[0,:] - dt_obs[0,:])**2) / (len(temp[0,:])))
sqrt(sum((temp[49,:] - dt_obs[49,:])**2) / (len(temp[49,:])))

# heatmap of temps  
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(dt_obs, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 35)
ax.set_ylabel("Depth", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Observed Temperature")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
plt.show()

