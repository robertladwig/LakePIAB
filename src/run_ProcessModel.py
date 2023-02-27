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
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel, run_thermalmodel, heating_module, diffusion_module, mixing_module, convection_module, ice_module

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
total_runtime =  365 *2 # 14 * 365
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
    area = hyps_all[0],
    volume = hyps_all[2],
    depth = hyps_all[1],
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
    diffusion_method = 'hondzoStefan',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
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
plt.subplots(figsize=(40,40))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of temps  
plt.subplots(figsize=(40,40))
sns.heatmap(temp_diff, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of diffusivities  
plt.subplots(figsize=(40,40))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()


# save model output

# initial temp.
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp_initial)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_temp_initial00.csv', index=None)

# heat temp.
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp_heat)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_temp_heat01.csv', index=None)

# diffusion temp.
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp_diff)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_temp_diff02.csv', index=None)

# mixing temp.
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp_mix)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_temp_mix03.csv', index=None)

# convection temp.
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp_conv)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_temp_conv04.csv', index=None)

# ice temp.
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(temp_ice)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_temp_total05.csv', index=None)

# diffusivity
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(diff)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_diff.csv', index=None)

# buoyancy
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(buoyancy)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_buoyancy.csv', index=None)

# meteorology
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(meteo)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df2.columns = ["AirTemp_degC", "Longwave_Wm-2",
                  "Latent_Wm-2", "Sensible_Wm-2", "Shortwave_Wm-2",
                  "lightExtinct_m-1","ShearVelocity_mS-1", "ShearStress_Nm-2",
                  "Area_m2", "CC", 'ea', 'Jlw', 'Uw', 'Pa', 'RH', 'PP', 'IceSnowAttCoeff',
                  'iceFlag', 'icemovAvg', 'density_snow', 'ice_prior', 'snow_prior', 
                  'snowice_prior', 'rho_snow_prior', 'IceSnowAttCoeff_prior', 'iceFlag_prior',
                  'dt_iceon_avg_prior', 'icemovAvg_prior']
df = pd.concat([df1, df2], axis = 1)
df.to_csv('../output/py_meteorology_input.csv', index=None)

    
# ice-snow
df1 = pd.DataFrame(times)
df1.columns = ['time']
t1 = np.matrix(icethickness)
t1 = t1.getT()
df2 = pd.DataFrame(t1)
df2.columns = ['ice']
t1 = np.matrix(snowthickness)
t1 = t1.getT()
df3 = pd.DataFrame(t1)
df3.columns = ['snow']
t1 = np.matrix(snowicethickness)
t1 = t1.getT()
df4 = pd.DataFrame(t1)
df4.columns = ['snowice']
df = pd.concat([df1, df2, df3, df4], axis = 1)
df.to_csv('../output/py_icesnow.csv', index=None)

# observed data
             
infile2  ="https://pasta.lternet.edu/package/data/eml/knb-lter-ntl/130/30/63d0587cf326e83f57b054bf2ad0f7fe".strip() 
infile2  = infile2.replace("https://","http://")
                 
dt2 =pd.read_csv(infile2 
          ,skiprows=1
            ,sep=","  
                ,quotechar='"' 
            , names=[
                    "sampledate",     
                    "year4",     
                    "month",     
                    "daynum",     
                    "hour",     
                    "depth",     
                    "wtemp",     
                    "flag_wtemp"    ]
# data type checking is commented out because it may cause data
# loads to fail if the data contains inconsistent values. Uncomment 
# the following lines to enable data type checking
         
#            ,dtype={ 
#             'sampledate':'str' , 
#             'year4':'float' , 
#             'month':'float' , 
#             'daynum':'float' , 
#             'hour':'str' , 
#             'depth':'float' , 
#             'wtemp':'float' ,  
#             'flag_wtemp':'str'  
#        }
          ,parse_dates=[
                        'sampledate',
                        'hour',
                ] 
    )
# Coerce the data into the types specified in the metadata 
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below. 
dt2=dt2.assign(sampledate_datetime=pd.to_datetime(dt2.sampledate,errors='coerce')) 
dt2.year4=pd.to_numeric(dt2.year4,errors='coerce') 
dt2.month=pd.to_numeric(dt2.month,errors='coerce') 
dt2.daynum=pd.to_numeric(dt2.daynum,errors='coerce') 
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below. 
dt2=dt2.assign(hour_datetime=pd.to_datetime(dt2.hour,errors='coerce')) 
dt2.depth=pd.to_numeric(dt2.depth,errors='coerce') 
dt2.wtemp=pd.to_numeric(dt2.wtemp,errors='coerce')  
dt2.flag_wtemp=dt2.flag_wtemp.astype('category') 
      
print("Here is a description of the data frame dt2 and number of lines\n")
print(dt2.info())
print("--------------------\n\n")                
print("Here is a summary of numerical variables in the data frame dt2\n")
print(dt2.describe())
print("--------------------\n\n")                
                         
print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")                 

print(dt2.sampledate.describe())               
print("--------------------\n\n")
                    
print(dt2.year4.describe())               
print("--------------------\n\n")
                    
print(dt2.month.describe())               
print("--------------------\n\n")
                    
print(dt2.daynum.describe())               
print("--------------------\n\n")
                    
print(dt2.hour.describe())               
print("--------------------\n\n")
                    
print(dt2.depth.describe())               
print("--------------------\n\n")
                    
print(dt2.wtemp.describe())               
print("--------------------\n\n")
                    
print(dt2.flag_wtemp.describe())               
print("--------------------\n\n")
                    
dt2.columns   
                 


# Assume that your dataframe is called "df"
# First, sort the dataframe by date and depth
df = dt2.sort_values(['sampledate_datetime', 'hour', 'depth'])

df['hour_numeric'] = pd.to_numeric(df['hour'], downcast='integer')

df['new_hour'] = df.apply(lambda row: row['hour_numeric'] // 100 if row['hour_numeric'] // 100 >= 1 else row['hour_numeric'], axis=1)

# Create a boolean index indicating the rows with an hour of 24
index = df['new_hour'] == 24

# Drop the rows with an hour of 24
df.drop(df[df['new_hour'] == 24].index,inplace=True)

df['new_hour'] = df['new_hour'].apply(lambda x: str(x).zfill(2))

# Combine the 'date' and 'hour' columns into a single string in the format '%Y-%m-%d %H'
df['datetime_str'] = df['sampledate_datetime'].astype(str) + ' ' + df['new_hour'].astype(str)

# Convert the 'datetime_str' column to a datetime column
df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H')

print(df)

# Next, create a list of the unique dates in the dataframe
dates = df['datetime'].unique()

# Create an empty list to store the interpolated data
interpolated_data = []

# Iterate over the unique dates
for date in dates:
  # Select the rows for the current date
  df_date = df[df['datetime'] == date]

  # Extract the depth and values for the current date
  depth = df_date['depth'].values
  values = df_date['wtemp'].values

  # Use scipy's interp1d function to create a function for interpolating the values at different depths
  f = interp1d(depth, values, bounds_error=False, fill_value=-999)

  # Create a list of the desired depths for interpolation
  new_depths = [i for i in range(25)]

  # Use the interpolation function to get the interpolated values at the desired depths
  interpolated_values = f(new_depths)

  # Add the interpolated values and the current date to the interpolated_data list
  interpolated_data.append({'time': date, 'values': interpolated_values})

# Convert the interpolated_data list into a pandas dataframe
interpolated_df = pd.DataFrame(interpolated_data)

def create_columns(row):
    # Create a dictionary with keys from 0 to 4 and values from the 'values' array
    d = {i: row['values'][i] for i in range(25)}
    # Return the dictionary as a Series
    return pd.Series(d)

# Create the new columns
df_new = interpolated_df.apply(create_columns, axis=1)

# Concatenate the original dataframe with the new columns
df_result = pd.concat([interpolated_df, df_new], axis=1)

# Drop the 'values' column
df_result = df_result.drop(columns=['values'])

# View the resulting dataframe
print(df_result)

# Create a new dataframe with the 'times' array as the 'time' column
df_times = pd.DataFrame({'time': times})

# Merge the df_result dataframe with the df_times dataframe using an outer join
df_merged = pd.merge(df_result, df_times, on='time', how='outer')

# View the merged dataframe
print(df_merged)

dt = df_merged.sort_values('time')

print(dt)

dt_red =  dt[dt['time'] <= max(times)]

# heatmap of temps  
# Set the figure size
plt.figure(figsize=(10, 8))

# Create a heat map plot
sns.heatmap(dt.set_index('time'), cmap='Reds')

# Show the plot
plt.show()


dt_red.to_csv('../output/py_observed_temp.csv', index=None, na_rep='NULL')
dt_red.to_csv('../output/py_observed_temp.csv', index=None)
dt_red.to_csv('../output/py_observed_temp.csv', index=None, na_rep='-999')
