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
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel, run_thermalmodel, heating_module, diffusion_module, mixing_module, convection_module, ice_module, run_thermalmodel_specific

## lake configurations
zmax = 25 # maximum lake depth
nx = 25 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
hyps_all = get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx, nx = nx)
                            
## all input data
all_data = pd.read_csv("../MCL/01_data_processing/all_data_lake_modeling_in_time.csv")
                     
hydrodynamic_timestep = 24 * dt
total_runtime =  365 *2 # 14 * 365
startTime = 1#150 * 24 * 3600
endTime =  (startTime + total_runtime * hydrodynamic_timestep) - 1

nTotalSteps = int(total_runtime * hydrodynamic_timestep/ dt)

unique_datetimes = all_data['time'].unique()

nCol = len(unique_datetimes)
all_temp = np.full([nx, nCol], np.nan)

for idn, n in enumerate(unique_datetimes):
    
    check_time = unique_datetimes[idn]
    data = all_data[all_data['time'] == check_time]

    ## here we define our initial profile
    u_ini = data['temp_initial00']

    
    res = run_thermalmodel_specific(  
        u = deepcopy(np.array(u_ini)),
        startTime = 1, 
        endTime = 2,
        area = hyps_all[0],
        volume = hyps_all[2],
        depth = hyps_all[1],
        zmax = zmax,
        nx = nx,
        dt = dt,
        dx = dx,
        Tair = np.mean(data['AirTemp_degC']),
        Jsw = np.mean(data['Shortwave_Wm-2']),
        kd_light = np.mean(data['lightExtinct_m-1']),
        CC = np.mean(data['CC']),
        ea = np.mean(data['ea']),
        Jlw = np.mean(data['Jlw']),
        Uw = np.mean(data['Uw']),
        Pa = np.mean(data['Pa']), 
        RH = np.mean(data['RH']),
        PP = np.mean(data['PP']),
        ice = np.mean(data['iceFlag_prior']),
        Hi = np.mean(data['ice_prior']),
        Hs = np.mean(data['snow_prior']),
        Hsi = np.mean(data['snowice_prior']),
        iceT = np.mean(data['icemovAvg_prior']),
        supercooled = 0,
        diffusion_method = 'hondzoStefan',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
        scheme='implicit',
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
        dt_iceon_avg = np.mean(data['dt_iceon_avg_prior']),
        Hgeo=0.1, # geothermal heat 0.1
        KEice=0,
        Ice_min=0.1,
        pgdl_mode = 'on',
        rho_snow = np.mean(data['rho_snow_prior']))
    
    temp =  res['temp_array']
    all_temp[:, idn] = temp


# heatmap of temps  
plt.subplots(figsize=(140,80))
sns.heatmap(all_temp, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()