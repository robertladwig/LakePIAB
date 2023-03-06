import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

#os.chdir("/home/robert/Projects/LakePIAB/src")
os.chdir("C:/Users/ladwi/Documents/Projects/R/LakePIAB/src")
#from oneD_HeatMixing_Functions import get_hyp()sography, provide_meteorology, initial_profile, run_thermalmodel_v1, run_hybridmodel_heating, run_hybridmodel_mixing, run_thermalmodel_v2
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel, run_thermalmodel_specific #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 25 # maximum lake depth
nx = 25 * 2 # number of layers we will have
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
total_runtime =  365 * 2 # 14 * 365
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

Start = datetime.datetime.now()

    
res = run_thermalmodel(  
    u = deepcopy(u_ini),
    startTime = startTime, 
    endTime = ( startTime + total_runtime * hydrodynamic_timestep) - 1,
    area = hyps_all[0][:-1],
    volume = hyps_all[2][:-1],
    depth = hyps_all[1][:-1],
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    daily_meteo = meteo_all[0],
    secview = meteo_all[1],
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,
    diffusion_method = 'hendersonSellers',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme='implicit',
    kd_light = 0.4, # 0.4,
    denThresh=1e-3,
    albedo = 0.1,
    eps=0.97,
    emissivity=0.97,
    sigma=5.67e-8,
    sw_factor = 1.0,
    wind_factor = 1.0,
    p2=1,
    B=0.61,
    g=9.81,
    Cd = 0.0013, # momentum coeff (wind)
    meltP=1,
    dt_iceon_avg=0.8,
    Hgeo=0.1, # geothermal heat 0.1
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

# heatmap of temps  
plt.subplots(figsize=(140,80))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of diffusivities  
plt.subplots(figsize=(140,80))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

time_step = 210 * 24 
depth_plot = hyps_all[1][:-1]
fig=plt.figure()
plt.plot(temp_initial[:,time_step], depth_plot, color="black")
plt.plot(temp_heat[:,time_step], depth_plot,color="red")
plt.plot(temp_diff[:,time_step], depth_plot,color="yellow")
# plt.plot(temp_mix[:,time_step], hyps_all[1],color="orange")
plt.plot(temp_conv[:,time_step], depth_plot,color="green")
plt.plot(temp_ice[:,time_step], depth_plot,color="blue")
plt.gca().invert_yaxis()
plt.show()

fig=plt.figure()
plt.plot(diff[:,time_step], depth_plot, color="black")
plt.gca().invert_yaxis()
plt.show()


# compare to observed data

df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df_simulated = pd.concat([df1, df2], axis = 1)

df = pd.read_csv('../output/NTL_observed_temp.csv')
df['datetime'] 
df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H')

# surface
df_1m_observed = df[df['depth'] == 0]
df_1m_observed = df_1m_observed[df_1m_observed['datetime'] <= '2012-01-01 00:00:00']

df_1m_simulated = df_simulated.iloc[:, 1]
df1 = pd.DataFrame(times)
df1.columns = ['datetime']
t1 = np.matrix(df_1m_simulated)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df2.columns= ['wtemp']
df_1m_simulated = pd.concat([df1, df2], axis = 1)

fig=plt.figure()
ax = df_1m_observed.plot(x='datetime', y=['wtemp'], color="black", style = '.')
df_1m_simulated.plot(x='datetime', y=['wtemp'], color="blue", ax = ax)
plt.show()

# bottom
df_20m_observed = df[df['depth'] == 20]
df_20m_observed = df_20m_observed[df_20m_observed['datetime'] <= '2012-01-01 00:00:00']

df_20m_simulated = df_simulated.iloc[:, 40]
df1 = pd.DataFrame(times)
df1.columns = ['datetime']
t1 = np.matrix(df_20m_simulated)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df2.columns= ['wtemp']
df_20m_simulated = pd.concat([df1, df2], axis = 1)

fig=plt.figure()
ax = df_20m_observed.plot(x='datetime', y=['wtemp'], color="black", style = '.')
df_20m_simulated.plot(x='datetime', y=['wtemp'], color="blue", ax = ax)
plt.show()


# middle
df_20m_observed = df[df['depth'] == 15]
df_20m_observed = df_20m_observed[df_20m_observed['datetime'] <= '2012-01-01 00:00:00']

df_20m_simulated = df_simulated.iloc[:, 30]
df1 = pd.DataFrame(times)
df1.columns = ['datetime']
t1 = np.matrix(df_20m_simulated)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df2.columns= ['wtemp']
df_20m_simulated = pd.concat([df1, df2], axis = 1)

fig=plt.figure()
ax = df_20m_observed.plot(x='datetime', y=['wtemp'], color="black", style = '.')
df_20m_simulated.plot(x='datetime', y=['wtemp'], color="blue", ax = ax)
plt.show()

