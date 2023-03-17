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
# from oneD_HeatMixing_Functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel_v1, run_hybridmodel_heating, run_hybridmodel_mixing, run_thermalmodel_v2
from processBased_lakeModel_functions import get_hypsography, eddy_diffusivity_hendersonSellers, eddy_diffusivity_munkAnderson, eddy_diffusivity, calc_dens, provide_meteorology, initial_profile, run_thermalmodel, run_thermalmodel, heating_module, diffusion_module, mixing_module, convection_module, ice_module, run_thermalmodel_specific

## lake configurations
zmax = 25 # maximum lake depth
nx = 25 * 2# number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
hyps_all = get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx, nx = nx)
                            
## all input data
all_data = pd.read_csv("../MCL/01_data_processing/all_data_lake_modeling_in_time.csv")
                     
hydrodynamic_timestep = 24 * dt
total_runtime =  (365*4) * hydrodynamic_timestep/dt  #365 *1 # 14 * 365
startTime =   (0 + 365*10) * hydrodynamic_timestep/dt #150 * 24 * 3600
endTime =  (startTime + total_runtime)  # * hydrodynamic_timestep/dt) - 1

nTotalSteps = int(total_runtime * hydrodynamic_timestep/ dt)

unique_datetimes = all_data['time'].unique()

nCol = len(unique_datetimes)
all_temp = np.full([nx, nCol], np.nan)

check_time = unique_datetimes[0]
data = all_data[all_data['time'] == check_time]

## here we define our initial profile
u_ini = data['temp_initial00']

u = deepcopy(np.array(u_ini))
startTime = 1 
endTime = 2
area = hyps_all[0][:-1]
volume = hyps_all[2][:-1]
depth = hyps_all[1][:-1]
zmax = zmax
nx = nx
dt = dt
dx = dx
Tair = np.mean(data['AirTemp_degC'])
Jsw = np.mean(data['Shortwave_Wm-2'])
kd_light = np.mean(data['lightExtinct_m-1'])
CC = np.mean(data['CC'])
ea = np.mean(data['ea'])
Jlw = np.mean(data['Jlw'])
Uw = np.mean(data['Uw'])
Pa = np.mean(data['Pa']) 
RH = np.mean(data['RH'])
PP = np.mean(data['PP'])
ice = np.mean(data['iceFlag_prior'])
Hi = np.mean(data['ice_prior'])
Hs = np.mean(data['snow_prior'])
Hsi = np.mean(data['snowice_prior']) 
iceT = np.mean(data['icemovAvg_prior'])
supercooled = 0
diffusion_method = 'hendersonSellers'# 'hendersonSellers' 'munkAnderson' 'hondzoStefan'
scheme='implicit'
km = 1.4 * 10**(-7) # 4 * 10**(-6) 
weight_kz = 0.5
denThresh=1e-3
albedo = 0.1
eps=0.97
emissivity=0.97
sigma=5.67e-8
sw_factor = 1.0
wind_factor = 1.0
p2=1
B=0.61
g=9.81
Cd = 0.0013 # momentum coeff (wind)
meltP=1
dt_iceon_avg = np.mean(data['dt_iceon_avg_prior'])
Hgeo=0.1 # geothermal heat 0.1
KEice=0
Ice_min=0.1
pgdl_mode = 'on'
rho_snow = np.mean(data['rho_snow_prior'])
rho_ice = 910
rho_fw = 1000
rho_new_snow = 250
rho_max_snow = 450
K_ice = 2.1
Cw = 4.18E6
L_ice = 333500
kd_snow = 0.9
kd_ice = 0.7


for idn, n in enumerate(unique_datetimes):
    
    print(n)
    
    check_time = unique_datetimes[idn]
    data = all_data[all_data['time'] == check_time]
    
    # boundary data
    Tair = np.mean(data['AirTemp_degC'])
    Jsw = np.mean(data['Shortwave_Wm-2'])
    kd_light = np.mean(data['lightExtinct_m-1'])
    CC = np.mean(data['CC'])
    ea = np.mean(data['ea'])
    Jlw = np.mean(data['Jlw'])
    Uw = np.mean(data['Uw'])
    Pa = np.mean(data['Pa']) 
    RH = np.mean(data['RH'])
    PP = np.mean(data['PP'])

    
    
    un = deepcopy(u)

    dens_u_n2 = calc_dens(un)

    if 'kz' in locals():
        1+1
    else: 
        kz = u * 0.0
        
    if diffusion_method == 'hendersonSellers':
        kz = eddy_diffusivity_hendersonSellers(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw,  43.100948, u, kz, Cd, km, weight_kz) / 1
    elif diffusion_method == 'munkAnderson':
        kz = eddy_diffusivity_munkAnderson(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw,  43.100948, Cd, u, kz) / 1
    elif diffusion_method == 'hondzoStefan':
        kz = eddy_diffusivity(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, u, kz) / 86400
  
  ## (1) HEATING
    heating_res = heating_module(
        un = u,
        area = area,
        volume = volume,
        depth = depth, 
        nx = nx,
        dt = dt,
        dx = dx,
        ice = ice,
        kd_ice = kd_ice,
        Tair = Tair,
        CC = CC,
        ea = ea,
        Jsw = Jsw,
        Jlw = Jlw,
        Uw = Uw,
        Pa= Pa,
        RH = RH,
        kd_light = kd_light,
        Hi = Hi,
        rho_snow = rho_snow,
        Hs = Hs)
   
    u = heating_res['temp']
    IceSnowAttCoeff = heating_res['IceSnowAttCoeff']
  
  
    icethickness_prior = Hi
    snowthickness_prior = Hs
    snowicethickness_prior = Hsi
    rho_snow_prior = rho_snow
    IceSnowAttCoeff_prior = IceSnowAttCoeff
    ice_prior = ice
    dt_iceon_avg_prior = dt_iceon_avg
    iceT_prior = iceT
  
  ## (2) ICE AND SNOW
    ice_res = ice_module(
        un = u,
        dt = dt,
        dx = dx,
        area = area,
        Tair = Tair,
        CC = CC,
        ea = ea,
        Jsw = Jsw,
        Jlw = Jlw,
        Uw = Uw,
        Pa= Pa,
        RH = RH,
        PP = PP,
        IceSnowAttCoeff = IceSnowAttCoeff,
        ice = ice,
        dt_iceon_avg = dt_iceon_avg,
        iceT = iceT,
        supercooled = supercooled,
        rho_snow = rho_snow,
        Hi = Hi,
        Hsi = Hsi,
        Hs = Hs)
  
    u = ice_res['temp']
    Hi = ice_res['icethickness']
    Hs = ice_res['snowthickness']
    Hsi = ice_res['snowicethickness']
    ice = ice_res['iceFlag']
    iceT = ice_res['icemovAvg']
    supercooled = ice_res['supercooled']
    rho_snow = ice_res['density_snow']

  
  ## (3) DIFFUSION
    diffusion_res = diffusion_module(
        un = u,
        kzn = kz,
        Uw = Uw,
        depth= depth,
        dx = dx,
        area = area,
        dt = dt,
        nx = nx,
        ice = ice, 
        diffusion_method = diffusion_method,
        scheme = scheme)
      
    u = diffusion_res['temp']
    kz = diffusion_res['diffusivity']
  

  ## (4) CONVECTION
    convection_res = convection_module(
        un = u,
        nx = nx,
        volume = volume)
  
    u = convection_res['temp']
    
    all_temp[:, idn] = u



# heatmap of temps  
N_pts = 6

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(all_temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 35)
ax.set_ylabel("Depth", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Hybrid Temperature")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = unique_datetimes[xticks_ix]
nelement = len(unique_datetimes)//N_pts
time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
plt.show()