import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt, log, atan, sin, radians, nan
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
from ancillary_functions import calc_cc, buoyancy_freq, center_buoyancy
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from numba import jit
from scipy.linalg import solve_banded
from scipy.stats.stats import pearsonr

## function to calculate density from temperature
def calc_dens(wtemp):
    dens = (999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) +
      (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + 
      (6.536336 * 1e-9 * wtemp**5))
    return dens

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity(rho, depth, g, rho_0, ice, area, T, diff):
    km = 1.4 * 10**(-7)
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    if ice == True:
      ak = 0.000898
    else:
      ak = 0.00706 *( max(area)/1E6)**(0.56)
    
    kz = ak * (buoy)**(-0.43)
    
        
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    
    return(kz + km)

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity_hendersonSellers(rho, depth, g, rho_0, ice, area, U10, latitude, T, diff):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    km = 1.4 * 10**(-7)
    
    U2 = U10 * 10
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    
    w_star = 1.2 * 10**(-3) * U2
    k_star = 6.6 * (sin(radians(latitude)))**(1/2) * U2**(-1.84)
    
    
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    
    #breakpoint()
    Ri = (-1 + (1 + 40 * np.array(buoy) * k**2 * np.array(depth)**2 / 
               (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    kz = (k * w_star * np.array(depth)) / (Pr * (1 + 37 * np.array(Ri)**2)) * np.exp(-k_star * np.array(depth))
    
    if ice == True:
      ak = 0.000898
    else:
      ak = 0.00706 *( max(area)/1E6)**(0.56)
      
    if (np.mean(T) <= 5):
        kz = kz * 1000
    
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    
    # kz = ak * (buoy)**(-0.43)
    return(kz +  km)

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity_munkAnderson(rho, depth, g, rho_0, ice, area, U10, latitude, Cd, T, diff):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    km = 1.4 * 10**(-7)
    rho_a = 1.2
    alpha = 10/3
    beta = 3/2
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    U2 = U10
    
    if U2 < 2.2:
        Cd = 1.08 * U2**(-0.15)* 10**(-3)
    elif 2.2 <= U2 < 5.0:
        Cd = (0.771 + 0.858 * U2**(-0.15)) *10**(-3)
    elif 5.0 <= U2 < 8.0:
        Cd = (0.867 + 0.0667 * U2**(-0.15)) * 10**(-3)
    elif 8.0 <= U2 < 25:
        Cd = (1.2 + 0.025 * U2**(-0.15)) * 10**(-3)
    elif 25 <= U2 < 50:
        Cd = 0.073 * U2**(-0.15) * 10**(-3)
    
    w_star = sqrt(rho_a / rho[0] * Cd * U2**2)
    k_star = 0.51 * (sin(radians(latitude))) / U2
    
    
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy
    # (uf./(kappa*z_edge).*exp(-ks*z_edge)).^2; 
    s_wall = (w_star / (k * np.array(depth)) * np.exp(k_star * np.array(depth)))**2
    s_wall =w_star/ (k * np.array(depth) *np.array(rho))
    
    
    X_HS = np.array(buoy)/(s_wall**2 + s_bg + s_seiche)
    Ri=(-1+(1+40*X_HS)**0.5)/20
    
    #breakpoint()
    #Ri = (-1 + (1 + 40 * np.array(buoy) * k**2 * np.array(depth)**2 / 
    #           (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    f_HS = (1.0 / (1 + alpha * Ri)**beta)
    f_HS[Ri == 0] = 1
    
    kz = (k * w_star * np.array(depth)) * np.exp(-k_star * np.array(depth)) * f_HS
    
    # modify according to Ekman layer depth
    
    tau_w = rho_a * Cd * U2**2
    u_star = sqrt(tau_w / rho_0)
    H_ekman = 0.4 * u_star / f
    
    e_w = xi * sqrt(Cd) * U2
    W_eff = e_w / (xi * sqrt(Cd))
    kz_ekman = 1/f * (rho_a / rho_0 * Cd / kullenberg)**2 * W_eff**2
    
    #kz[depth < H_ekman] = kz_ekman 
    
    if (np.mean(T) <= 5):
        kz = kz * 1000
    
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    return(kz +  km)

def crank_nicholson(T, dz, dt, kappa, depth, area, volume):
    #breakpoint()
    N = len(T)
    T_new = T
    
    r = kappa * dt /dz**2 
    r = r[1:-1]
    
    A = np.zeros((3, N -2))
    
    A[0, :] = -r
    A[1,:] = 1 + 2*r
    A[2, :] = -r
    
    B = np.zeros(N -2)
    B = r*T[2:] + (1-2*r)*r*T[1:-1] + r*T[:-2]
    B[0] = B[0] + r[0]*T[0]
    B[-1] = B[-1] +r[-1]*T[-1]
    
    T_new[1:-1] = solve_banded((1,1), A, B)
    
def provide_meteorology(meteofile, secchifile, windfactor):

    meteo = pd.read_csv(meteofile)
    daily_meteo = meteo
    daily_meteo['date'] = pd.to_datetime(daily_meteo['datetime'])
    daily_meteo['Cloud_Cover'] = calc_cc(date = daily_meteo['date'],
                                                airt = daily_meteo['Air_Temperature_celsius'],
                                                relh = daily_meteo['Relative_Humidity_percent'],
                                                swr = daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'],
                                                lat = 43, lon = -89.41,
                                                elev = 258)
    daily_meteo['dt'] = (daily_meteo['date'] - daily_meteo['date'][0]).astype('timedelta64[s]') + 1
    daily_meteo['ea'] = (daily_meteo['Relative_Humidity_percent'] * 
      (4.596 * np.exp((17.27*(daily_meteo['Air_Temperature_celsius'])) /
      (237.3 + (daily_meteo['Air_Temperature_celsius']) ))) / 100)
    daily_meteo['ea'] = ((101.325 * np.exp(13.3185 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15))) -
      1.976 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**2 -
      0.6445 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**3 -
      0.1229 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**4)) * daily_meteo['Relative_Humidity_percent']/100)
    daily_meteo['ea'] = (daily_meteo['Relative_Humidity_percent']/100) * 10**(9.28603523 - 2322.37885/(daily_meteo['Air_Temperature_celsius'] + 273.15))
    startDate = pd.to_datetime(daily_meteo.loc[0, 'date']) 
    
    ## calibration parameters
    daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] = daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] 
    daily_meteo['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] = daily_meteo['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] * windfactor # wind speed multiplier
    
    date_time = daily_meteo.date

    daily_meteo['day_of_year_list'] = [t.timetuple().tm_yday for t in date_time]
    daily_meteo['time_of_day_list'] = [t.hour for t in date_time]
    ## light
    # Package ID: knb-lter-ntl.31.30 Cataloging System:https://pasta.edirepository.org.
    # Data set title: North Temperate Lakes LTER: Secchi Disk Depth; Other Auxiliary Base Crew Sample Data 1981 - current.
    if secchifile is not None:
        secview0 = pd.read_csv(secchifile)
        secview0['sampledate'] = pd.to_datetime(secview0['sampledate'])
        secview = secview0.loc[secview0['sampledate'] >= startDate]
        if secview['sampledate'].min() >= startDate:
          firstVal = secview.loc[secview['sampledate'] == secview['sampledate'].min(), 'secnview'].values[0]
          firstRow = pd.DataFrame(data={'sampledate': [startDate], 'secnview':[firstVal]})
          secview = pd.concat([firstRow, secview], ignore_index=True)
      
          
        secview['dt'] = (secview['sampledate'] - secview['sampledate'][0]).astype('timedelta64[s]') + 1
        secview['kd'] = 1.7 / secview['secnview']
        secview['kd'] = secview.set_index('sampledate')['kd'].interpolate(method="linear").values
    else:
        secview = None
    
    return([daily_meteo, secview])
  
def initial_profile(initfile, nx, dx, depth, startDate):
  #meteo = processed_meteo
  #startDate = meteo['date'].min()
  obs = pd.read_csv(initfile)
  obs['datetime'] = pd.to_datetime(obs['datetime'])
  obs['ditt'] = abs(obs['datetime'] - startDate)
  init_df = obs.loc[obs['ditt'] == obs['ditt'].min()]
  if max(depth) > init_df.Depth_meter.max():
    lastRow = init_df.loc[init_df.Depth_meter == init_df.Depth_meter.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'Depth_meter'] = max(depth)
    
  profile_fun = interp1d(init_df.Depth_meter.values, init_df.Water_Temperature_celsius.values)
  out_depths = np.linspace(0, nx*dx, nx) # these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  u = profile_fun(out_depths)
  
  # TODO implement warning about profile vs. met start date
  
  return(u)

def get_hypsography(hypsofile, dx, nx):
  hyps = pd.read_csv(hypsofile)
  out_depths = np.linspace(1, nx*dx, nx)
  area_fun = interp1d(hyps.Depth_meter.values, hyps.Area_meterSquared.values)
  area = area_fun(out_depths)
  area[-1] = area[-2] - 1 # TODO: confirm this is correct
  depth = np.linspace(0, nx*dx, nx)
  
  volume = 0.5 * (area[:-1] + area[1:]) * np.diff(depth)
  volume = np.append(volume, 1000)
  
  return([area, depth, volume])

def longwave(cc, sigma, Tair, ea, emissivity, Jlw):  # longwave radiation into
  Tair = Tair + 273.15
  p = (1.33 * ea/Tair)
  Ea = 1.24 * (1 + 0.17 * cc**2) * p**(1/7)
  lw = emissivity * Ea *sigma * Tair**4
  return(lw)

def backscattering(emissivity, sigma, Twater, eps): # backscattering longwave 
  # radiation from the lake
  Twater = Twater + 273.15
  back = -1 * (eps * sigma * (Twater)**4) 
  return(back)

def PSIM(zeta):
  # Function to compute stability functions for momentum
  if zeta < 0.0:
    X = (1 - 16*zeta)**0.25
    psim = 2*log((1 + X)/2) + log((1 + X*X)/2)-2*atan(X) + pi/2 
  elif zeta > 0.0:
    if zeta > 0.5:
      if zeta > 10.0:
        psim = log(zeta) - 0.76*zeta - 12.093
      else:
        psim = 0.5/(zeta*zeta) - 4.25/zeta - 7.0*log(zeta) - 0.852
    else:
      psim = -5*zeta
  # Stable case
  else:
    psim = 0.0
  return(psim)

def PSITE(zeta):
  # Function to compute stability functions for sensible and latent heat
  if zeta < 0.0:
    X = (1 - 16*zeta)**0.25
    psite = 2*log((1 + X*X)/2)
  elif zeta > 0.0:# Stable case
    if zeta > 0.5:
      if zeta > 10.0:
        psite = log(zeta) - 0.76*zeta - 12.093
      else:
        psite = 0.5/(zeta*zeta) - 4.25/zeta - 7.0*log(zeta) - 0.852
    else: 
      psite = -5*zeta
  else:
    psite = 0.0
  return(psite)

# def sensible(p2, B, Tair, Twater, Uw): # convection / sensible heat
#   Twater = Twater + 273.15
#   Tair = Tair + 273.15
#   fu = 4.4 + 1.82 * Uw + 0.26 *(Twater - Tair)
#   sensible = -1 * ( p2 * B * fu * (Twater - Tair)) 
#   return(sensible)

def sensible(Tair, Twater, Uw, p2, pa, ea, RH, A, Cd = 0.0013): # evaporation / latent heat
  global H
  # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012839
  
  # Tair =0
  # Twater = 0
  # Uw = 0.01
  # pa = 98393
  # ea = 6.079572
  # A = 31861
  # Cd = 0.0037
  
  const_SpecificHeatAir = 1005;           # Units : J kg-1 K-1
  const_vonKarman = 0.41;                 # Units : none
  const_Gravity = 9.81;                   # Units : m s-2
  const_Charnock = Cd;   
  
  U_Z = Uw
  if Uw <= 0:
    U_Z = 1e-3
  T = Tair
  if Tair == 0:
    T = np.random.uniform(low = 1e-7, high = 1e-5)
  T0 = Twater
  if Twater == 0: 
    T0 = np.random.uniform(low = 1e-7, high = 1e-5)
  Rh = RH
  p = pa/100
  z = 2
  
  # Step 2c - Compute saturated vapour pressure at air temperature
  e_s = 6.11*exp(17.27*T/(237.3+T)) # Units : mb ##REF##
  # Step 2d - Compute vapour pressure
  e_a = Rh*e_s/100 # Units : mb
  ### End step 2
  
  ### Step 3 - Compute other values used in flux calculations
  # Step 3a - Compute specific humidity
  q_z = 0.622*e_a/p # Units: kg kg-1
  # Step 3b - Compute saturated vapour pressure at water temperature
  e_sat = 6.11*exp(17.27*T0/(237.3+T0)) # Units : mb ##REF##
  # Step 3c - Compute humidity at saturation (Henderson-Sellers 1986 eqn 36)
  q_s = 0.622*e_sat/p # Units: kg kg-1
  # Step 3d - Compute latent heat of vaporisation
  L_v = 2.501e6-2370*T0 # Units : J kg-1 ** EQUATION FROM PIET ##REF##
  # Step 3e - Compute gas constant for moist air
  R_a = 287*(1+0.608*q_z) # Units : J kg-1 K-1
  # Step 3f - Compute air density
  rho_a = 100*p/(R_a*(T+273.16)) # Units : kg m-3
  # Step 3g - Compute kinematic viscosity of air 
  v = (1./rho_a)*(4.94e-8*T + 1.7184e-5) # Units : m2 s-1
  # Step 3h - Compute virtual air temperature and virtual air-water temperature difference
  T_v = (T+273.16)*(1+0.61*q_z) # Units - K
  T_ov = (T0+273.16)*(1+0.61*q_s) # Units - K
  del_theta = T_ov - T_v
  # Step 3h - Compute water density 
  rho_w = 1000*(1-1.9549*0.00001*abs(T0-3.84)**1.68)
  ### End step 3
  
  # step 4
  u_star = U_Z *sqrt(0.00104+0.0015/(1+exp((-U_Z+12.5)/1.56))) # Amorocho and DeVries, initialise ustar using U_Z
  
  if u_star == 0: 
    u_star = 1e-6
  
  z_0 = (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
  z_0_prev=z_0*1.1 # To initiate the iteration
  

  
  while (abs((z_0 - z_0_prev))/abs(z_0_prev) > 0.000001): # Converge when z_0 within 0.0001# of previous value 
    u_star=const_vonKarman*U_Z/(log(z/z_0))  # Compute u_star
    dummy = z_0 # Used to control while loop
    z_0=(const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star); # Compute new roughness length
    z_0_prev = dummy # Used to control while loop
  
  # Step 4d - Compute initial neutral drag coefficient
  C_DN = (u_star**2)/(U_Z**2) # Units - none
  # Step 4e - Compute roughness Reynolds number 
  Re_star = u_star*z_0/v # Units - none
  # Step 4f - Compute initial roughness length for temperature
  z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57) # Units - m
  z_T = z_T.real # Get real components, and NaN can create imag component despite no data
  # Step 4g - Compute initial roughness length for vapour 
  z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57); # Units - m 
  z_E = z_E.real # Get real components, and NaN can create imag component despite no data
  # Step 4h - Compute initial neutral sensible heat transfer coefficient 
  C_HN = const_vonKarman*sqrt(C_DN)/(log(z/z_T)) 
  # Step 4i - Compute initial neutral latent heat transfer coefficient
  C_EN = const_vonKarman*sqrt(C_DN)/(log(z/z_E))
  ### End step 4
  
  ### Step 5 - Start iteration to compute corrections for atmospheric stability
  # for (i1 in 1:length(U_Z)){

  # Step 5a - Compute initial sensible heat flux based on neutral coefficients
  H_initial = rho_a*const_SpecificHeatAir*C_HN*U_Z*(T0-T) # Units : W m-2
  # Step 5b - Compute initial latent heat flux based on neutral coefficients
  E_initial = rho_a*L_v*C_EN*U_Z*(q_s-q_z) # Units : W m-2
  # Step 5c - Compute initial Monin-Obukhov length
  L_initial = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H_initial/const_SpecificHeatAir + 0.61*E_initial*(T+273.16)/L_v)) # Units - m
  # Step 5d - Compute initial stability parameter
  zeta_initial = z/L_initial
  # Step 5e - Compute initial stability function
  psim=PSIM(zeta_initial) # Momentum stability function
  psit=PSITE(zeta_initial) # Sensible heat stability function
  psie=PSITE(zeta_initial) # Latent heat stability function
  # Step 5f - Compute corrected coefficients
  C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2
  C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
  C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
  # Step 5g - Start iteration
  L_prev = L_initial
  L = L_prev*1.1 # Initialise while loop
  count=np.zeros(1);
  while (abs((L - L_prev))/abs(L_prev) > 0.000001):
    # Iteration counter
    count=count+1;
    if count > 20:
      break
    # Step 5i - Compute new z_O, roughness length for momentum
    z_0= (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
    # Step 5j - Compute new Re_star
    Re_star = u_star*z_0/v
    # Step 5k - Compute new z_T, roughness length for temperature
    z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5l - Compute new z_E, roughness length for vapour
    z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5p - Compute new stability parameter
    zeta = z/L;
    #fprintf('zeta #g\n',zeta);
    # Step 5q - Check and enforce bounds on zeta
    if zeta > 15:
      zeta = 15
    elif zeta < -15 :
      zeta = -15
    # Step 5r - Compute new stability functions
    psim=PSIM(zeta) # Momentum stability function
    psit=PSITE(zeta) # Sensible heat stability function
    psie=PSITE(zeta) # Latent heat stability function
    # Step 5s - Compute corrected coefficients
    C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2;
    C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
    C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
    # Step 5m - Compute new H (now using corrected coefficients)
    H = rho_a*const_SpecificHeatAir*C_H*U_Z*(T0-T);
    # Step 5n - Compute new E (now using corrected coefficients)
    E = rho_a*L_v*C_E*U_Z*(q_s-q_z);
    # Step 5h - Compute new u_star
    u_star=sqrt(C_D*U_Z**2);
    # Step 5o - Compute new Monin-Obukhov length
    dummy = L; # Used to control while loop
    L = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H/const_SpecificHeatAir + 0.61*E*(T+273.16)/L_v));
    L_prev = dummy; # Used to control while loop

  
  sensible = H
  return sensible* (-1)

# def latent(Tair, Twater, Uw, p2, pa, ea, RH): # evaporation / latent heat
#   Twater = Twater + 273.15
#   Tair = Tair + 273.15
#   Pressure = pa / 100
#   fu = 4.4 + 1.82 * Uw + 0.26 *(Twater - Tair)
#   fw = 0.61 * (1 + 10**(-6) * Pressure * (4.5 + 6 * 10**(-5) * Twater**2))
#   ew = fw * 10 * ((0.7859+0.03477* Twater)/(1+0.00412* Twater))
#   latent = -1* fu * p2 * (ew - ea)# * 1.33) // * 1/6
#   return(latent)
def latent(Tair, Twater, Uw, p2, pa, ea, RH, A, Cd = 0.0013): # evaporation / latent heat
  global E
  # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012839
   
  # Tair =0
  # Twater = 0
  # Uw = 0.01
  # pa = 98393
  # ea = 6.079572
  # A = 31861
  # Cd = 0.0037
  
  const_SpecificHeatAir = 1005;           # Units : J kg-1 K-1
  const_vonKarman = 0.41;                 # Units : none
  const_Gravity = 9.81;                   # Units : m s-2
  const_Charnock = Cd;   
  
  U_Z = Uw
  if Uw <= 0:
    U_Z = 1e-3
  T = Tair
  if Tair == 0:
    T = np.random.uniform(low = 1e-7, high = 1e-5)
  T0 = Twater
  if Twater == 0: 
    T0 = np.random.uniform(low = 1e-7, high = 1e-5)
  Rh = RH
  p = pa/100
  z = 2
  
  # Step 2c - Compute saturated vapour pressure at air temperature
  e_s = 6.11*exp(17.27*T/(237.3+T)) # Units : mb ##REF##
  # Step 2d - Compute vapour pressure
  e_a = Rh*e_s/100 # Units : mb
  ### End step 2
  
  ### Step 3 - Compute other values used in flux calculations
  # Step 3a - Compute specific humidity
  q_z = 0.622*e_a/p # Units: kg kg-1
  # Step 3b - Compute saturated vapour pressure at water temperature
  e_sat = 6.11*exp(17.27*T0/(237.3+T0)) # Units : mb ##REF##
  # Step 3c - Compute humidity at saturation (Henderson-Sellers 1986 eqn 36)
  q_s = 0.622*e_sat/p # Units: kg kg-1
  # Step 3d - Compute latent heat of vaporisation
  L_v = 2.501e6-2370*T0 # Units : J kg-1 ** EQUATION FROM PIET ##REF##
  # Step 3e - Compute gas constant for moist air
  R_a = 287*(1+0.608*q_z) # Units : J kg-1 K-1
  # Step 3f - Compute air density
  rho_a = 100*p/(R_a*(T+273.16)) # Units : kg m-3
  # Step 3g - Compute kinematic viscosity of air 
  v = (1./rho_a)*(4.94e-8*T + 1.7184e-5) # Units : m2 s-1
  # Step 3h - Compute virtual air temperature and virtual air-water temperature difference
  T_v = (T+273.16)*(1+0.61*q_z) # Units - K
  T_ov = (T0+273.16)*(1+0.61*q_s) # Units - K
  del_theta = T_ov - T_v
  # Step 3h - Compute water density 
  rho_w = 1000*(1-1.9549*0.00001*abs(T0-3.84)**1.68)
  ### End step 3
  
  # step 4
  u_star = U_Z *sqrt(0.00104+0.0015/(1+exp((-U_Z+12.5)/1.56))) # Amorocho and DeVries, initialise ustar using U_Z
  
  if u_star == 0: 
    u_star = 1e-6
  
  z_0 = (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
  z_0_prev=z_0*1.1 # To initiate the iteration
  

  
  while (abs((z_0 - z_0_prev))/abs(z_0_prev) > 0.000001): # Converge when z_0 within 0.0001# of previous value 
    u_star=const_vonKarman*U_Z/(log(z/z_0))  # Compute u_star
    dummy = z_0 # Used to control while loop
    z_0=(const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star); # Compute new roughness length
    z_0_prev = dummy # Used to control while loop
  
  # Step 4d - Compute initial neutral drag coefficient
  C_DN = (u_star**2)/(U_Z**2) # Units - none
  # Step 4e - Compute roughness Reynolds number 
  Re_star = u_star*z_0/v # Units - none
  # Step 4f - Compute initial roughness length for temperature
  z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57) # Units - m
  z_T = z_T.real # Get real components, and NaN can create imag component despite no data
  # Step 4g - Compute initial roughness length for vapour 
  z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57); # Units - m 
  z_E = z_E.real # Get real components, and NaN can create imag component despite no data
  # Step 4h - Compute initial neutral sensible heat transfer coefficient 
  C_HN = const_vonKarman*sqrt(C_DN)/(log(z/z_T)) 
  # Step 4i - Compute initial neutral latent heat transfer coefficient
  C_EN = const_vonKarman*sqrt(C_DN)/(log(z/z_E))
  ### End step 4
  
  ### Step 5 - Start iteration to compute corrections for atmospheric stability
  # for (i1 in 1:length(U_Z)){

  # Step 5a - Compute initial sensible heat flux based on neutral coefficients
  H_initial = rho_a*const_SpecificHeatAir*C_HN*U_Z*(T0-T) # Units : W m-2
  # Step 5b - Compute initial latent heat flux based on neutral coefficients
  E_initial = rho_a*L_v*C_EN*U_Z*(q_s-q_z) # Units : W m-2
  # Step 5c - Compute initial Monin-Obukhov length
  L_initial = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H_initial/const_SpecificHeatAir + 0.61*E_initial*(T+273.16)/L_v)) # Units - m
  # Step 5d - Compute initial stability parameter
  zeta_initial = z/L_initial
  # Step 5e - Compute initial stability function
  psim=PSIM(zeta_initial) # Momentum stability function
  psit=PSITE(zeta_initial) # Sensible heat stability function
  psie=PSITE(zeta_initial) # Latent heat stability function
  # Step 5f - Compute corrected coefficients
  C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2
  C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
  C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
  # Step 5g - Start iteration
  L_prev = L_initial
  L = L_prev*1.1 # Initialise while loop
  count=np.zeros(1);
  while (abs((L - L_prev))/abs(L_prev) > 0.000001):
    # Iteration counter
    count=count+1;
    if count > 20:
      break
    # Step 5i - Compute new z_O, roughness length for momentum
    z_0= (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
    # Step 5j - Compute new Re_star
    Re_star = u_star*z_0/v
    # Step 5k - Compute new z_T, roughness length for temperature
    z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5l - Compute new z_E, roughness length for vapour
    z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5p - Compute new stability parameter
    zeta = z/L;
    #fprintf('zeta #g\n',zeta);
    # Step 5q - Check and enforce bounds on zeta
    if zeta > 15:
      zeta = 15
    elif zeta < -15 :
      zeta = -15
    # Step 5r - Compute new stability functions
    psim=PSIM(zeta) # Momentum stability function
    psit=PSITE(zeta) # Sensible heat stability function
    psie=PSITE(zeta) # Latent heat stability function
    # Step 5s - Compute corrected coefficients
    C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2;
    C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
    C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
    # Step 5m - Compute new H (now using corrected coefficients)
    H = rho_a*const_SpecificHeatAir*C_H*U_Z*(T0-T);
    # Step 5n - Compute new E (now using corrected coefficients)
    E = rho_a*L_v*C_E*U_Z*(q_s-q_z);
    # Step 5h - Compute new u_star
    u_star=sqrt(C_D*U_Z**2);
    # Step 5o - Compute new Monin-Obukhov length
    dummy = L; # Used to control while loop
    L = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H/const_SpecificHeatAir + 0.61*E*(T+273.16)/L_v));
    L_prev = dummy; # Used to control while loop
  # Converge when L within 0.0001# or previous L
    
  # Need to iterate separately for each record
  
  
  ### End step 5
  
  # Take real values to remove any complex values that arise from missing data or NaN.
  # C_D=C_D.real 
  # C_E=C_E.real 
  # C_H=C_H.real 
  # z_0=z_0.real 
  # z_E=z_E.real 
  # z_T=z_T.real
  
  # Compute evaporation [mm/day]
  Evap = 86400*1000*E/(rho_w*L_v)
  
  latent = E
  return latent* (-1)

#@jit(nopython=True)
def run_thermalmodel_v1(
  u, 
  startTime, 
  endTime,
  area,
  volume,
  depth,
  zmax,
  nx,
  dt,
  dx,
  daily_meteo,
  secview,
  ice=False,
  Hi=0,
  iceT=6,
  supercooled=0,
  scheme='implicit',
  kd_light=None,
  denThresh=1e-3,
  albedo=0.1,
  eps=0.97,
  emissivity=0.97,
  sigma=5.67e-8,
  sw_factor = 1.0,
  wind_factor = 1.0,
  p2=1,
  B=0.61,
  g=9.81,
  Cd=0.0013, # momentum coeff (wind)
  meltP=1,
  dt_iceon_avg=0.8,
  Hgeo=0.1, # geothermal heat
  KEice=1/1000,
  Ice_min=0.1,
  pgdl_mode='on',
  Hs = 0,
  rho_snow = 250,
  Hsi = 0,
  rho_ice = 910,
  rho_fw = 1000,
  rho_new_snow = 250,
  rho_max_snow = 450,
  K_ice = 2.1,
  Cw = 4.18E6,
  L_ice = 333500,
  kd_snow = 0.9,
  kd_ice = 0.7):
    

  
  ## linearization of driver data, so model can have dynamic step
  Jsw_fillvals = tuple(daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values[[0, -1]])
  Jsw = interp1d(daily_meteo.dt.values, daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jsw_fillvals, bounds_error=False)
  Jlw_fillvals = tuple(daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values[[0,-1]])
  Jlw = interp1d(daily_meteo.dt.values, daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jlw_fillvals, bounds_error=False)
  Tair_fillvals = tuple(daily_meteo.Air_Temperature_celsius.values[[0,-1]])
  Tair = interp1d(daily_meteo.dt.values, daily_meteo.Air_Temperature_celsius.values, kind = "linear", fill_value=Tair_fillvals, bounds_error=False)
  ea_fillvals = tuple(daily_meteo.ea.values[[0,-1]])
  ea = interp1d(daily_meteo.dt.values, daily_meteo.ea.values, kind = "linear", fill_value=ea_fillvals, bounds_error=False)
  Uw_fillvals = tuple(daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values[[0, -1]])
  Uw = interp1d(daily_meteo.dt.values, wind_factor * daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values, kind = "linear", fill_value=Uw_fillvals, bounds_error=False)
  CC_fillvals = tuple(daily_meteo.Cloud_Cover.values[[0,-1]])
  CC = interp1d(daily_meteo.dt.values, daily_meteo.Cloud_Cover.values, kind = "linear", fill_value=CC_fillvals, bounds_error=False)
  Pa_fillvals = tuple(daily_meteo.Surface_Level_Barometric_Pressure_pascal.values[[0,-1]])
  Pa = interp1d(daily_meteo.dt.values, daily_meteo.Surface_Level_Barometric_Pressure_pascal.values, kind = "linear", fill_value=Pa_fillvals, bounds_error=False)
  if kd_light is None:
      kd_fillvals = tuple(secview.kd.values[[0,-1]])
      kd = interp1d(secview.dt.values, secview.kd.values, kind = "linear", fill_value=kd_fillvals, bounds_error=False)
  RH_fillvals = tuple(daily_meteo.Relative_Humidity_percent.values[[0,-1]])
  RH = interp1d(daily_meteo.dt.values, daily_meteo.Relative_Humidity_percent.values, kind = "linear", fill_value=RH_fillvals, bounds_error=False)
  PP_fillvals = tuple(daily_meteo.Precipitation_millimeterPerDay.values[[0,-1]])
  PP = interp1d(daily_meteo.dt.values, daily_meteo.Precipitation_millimeterPerDay.values, kind = "linear", fill_value=PP_fillvals, bounds_error=False)

  #plt.plot(PP(np.arange(1,1e7,1)))
  #plt.plot(daily_meteo.Precipitation_millimeterPerDay.values[[0, -1]])
  
  step_times = np.arange(startTime, endTime, dt)
  nCol = len(step_times)
  um = np.full([nx, nCol], np.nan)
  kzm = np.full([nx, nCol], np.nan)
  n2m = np.full([(nx-1), nCol], np.nan)
  mix = np.full([1,nCol], np.nan)
  therm_z = np.full([1,nCol], np.nan)
  mix_z = np.full([1,nCol], np.nan)
  Him = np.full([1,nCol], np.nan)
  Hm = np.full([nx, nCol], np.nan) 
  Qm = np.full([1,nCol], np.nan)
  Him= np.full([1,nCol], np.nan)
  Hsm= np.full([1,nCol], np.nan)
  Hsim= np.full([1,nCol], np.nan)
  Ticem= np.full([1,nCol], np.nan)
  
  if pgdl_mode == 'on':
    um_initial = np.full([nx, nCol], np.nan)
    um_heat = np.full([nx, nCol], np.nan)
    um_diff = np.full([nx, nCol], np.nan)
    um_mix = np.full([nx, nCol], np.nan)
    um_conv = np.full([nx, nCol], np.nan)
    um_ice = np.full([nx, nCol], np.nan)
    n2_pgdl = np.full([nx, nCol], np.nan)
    meteo_pgdl = np.full([9, nCol], np.nan)
  
  if not kd_light is None:
    def kd(n): # using this shortcut for now / testing if it works
      return kd_light

  

  times = np.arange(startTime, endTime, dt)
  for idn, n in enumerate(times):
    
    un = deepcopy(u)
    dens_u_n2 = calc_dens(u)
    time_ind = np.where(times == n)
    
    if pgdl_mode == 'on':
      n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
      n2_pgdl[:,idn] = np.concatenate([n2, np.array([np.nan])])
      um_initial[:, idn] = u
     
    if 'kzn' in locals():
        1+1
    else: 
        kzn = u * 0.0
    
    kz = eddy_diffusivity(dens_u_n2, depth, 9.81, np.mean(dens_u_n2) , ice, area, u, diff = kzn) / 86400
    

    if ice and Tair(n) <= 0:
      kzn = kz
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif (ice and Tair(n) >= 0):
      kzn = kz
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif not ice:
      kzn = kz
      albedo = 0.1
      IceSnowAttCoeff = 1
    kzm[:,idn] = kzn
    
    ## (1) HEAT ADDITION
    # surface heat flux
    start_time = datetime.datetime.now()
    Q = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd))  
    
    # heat addition over depth
    
    
    if ice:
        H =  IceSnowAttCoeff * (Jsw(n) * sw_factor)  * np.exp(-(kd_light) * depth)
    else:
        H =  (1- albedo) * (Jsw(n) * sw_factor)  * np.exp(-(kd_light ) * depth)
    
    Hg = (area[:-1]-area[1:])/dx * Hgeo/(4181 * calc_dens(un[0]))
    Hg = np.append(Hg, Hg.min())
    
    u[0] = (un[0] + 
        (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
        Hg[0]) * dt/area[0])
      # all layers in between
    for i in range(1,(nx-1)):
         u[i] = un[i] + (abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i]
      # bottom layer
    u[(nx-1)] = un[(nx-1)] + (abs(H[(nx-1)]-H[(nx-2)]) * area[(nx-1)]/(area[(nx-1)] * dx) * 1/(4181 * calc_dens(un[(nx-1)])) +Hg[(nx-1)]/area[(nx-1)]) * dt

    if pgdl_mode == 'on':
      um_heat[:, idn] = u
      Hm[:, idn] = H
      Qm[0, idn] = Q
    
    end_time = datetime.datetime.now()
    print("heating: " + str(end_time - start_time))


    ## (2) DIFFUSION
    start_time = datetime.datetime.now()
    if scheme == 'implicit':

      
        # IMPLEMENTATION OF CRANK-NICHOLSON SCHEME
        
        j = len(un)
        y = np.zeros((len(un), len(un)))
        
        alpha = (dt/dx**2) * kzn
        
        az = - alpha # subdiagonal
        bz = 2 * (1 + alpha) # diagonal
        cz = - alpha # superdiagonal
        
        bz[0] = 1
        # az[len(az)-2] = 0
        bz[len(bz)-1] = 1
        cz[0] = 0
        
        az =  np.delete(az,0)
        cz =  np.delete(cz,len(cz)-1)
        
        # tridiagonal matrix
        for k in range(j-1):
            y[k][k] = bz[k]
            y[k][k+1] = cz[k]
            y[k+1][k] = az[k]
        
        # y[0,1] = 0    
        # y[j-1, j-1] = 1
        y[j-1, j-2] = 0
        y[j-1, j-1] = 1
        
        # print(y[0:4])
        
        mn = un * 0.0    
        mn[0] = u[0]
        mn[len(mn)-1] = u[len(u)-1]
        
        for k in range(1,j-2):
            mn[k] = alpha[k] * u[k-1] + 2 * (1 - alpha[k]) * u[k] + alpha[k] * u[k+1]

    # DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE

        u = np.linalg.solve(y, mn)

        
    # TODO: implement / figure out this
    if scheme == 'explicit':
      u[0] = (un[0] + 
        (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
        Hg[0]) * dt/area[0])
      # all layers in between
      for i in range(1,(nx-1)):
        u[i] = (un[i] + (area[i] * kzn[i] * 1 / dx**2 * (un[i+1] - 2 * un[i] + un[i-1]) +
          abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i])
      # bottom layer
      u[(nx-1)] = (un[(nx-1)] +
      (abs(H[(nx-1)]-H[(nx-1)-1]) * area[(nx-1)]/(area[(nx-1)]*dx) * 1/(4181 * calc_dens(un[(nx-1)])) +
      Hg[(nx-1)]/area[(nx-1)]) * dt)
                                                           
    if pgdl_mode == 'on':
      um_diff[:, idn] = u

    end_time = datetime.datetime.now()
    print("diffusion: " + str(end_time - start_time))
      
    ## (3) TURBULENT MIXING OF MIXED LAYER
    # the mixed layer depth is determined for each time step by comparing kinetic 
    # energy available from wind and the potential energy required to completely 
    # mix the water column to a given depth
    start_time = datetime.datetime.now()
    Zcv = np.sum(depth * area) / sum(area)  # center of volume
    tau = 1.225 * Cd * Uw(n) ** 2 # wind shear is air density times wind velocity 
    if (Uw(n) <= 15):
      c10 = 0.0005 * sqrt(Uw(n))
    else:
      c10 = 0.0026
      
    shear = sqrt((c10 * calc_dens(un[0]))/1.225) *  Uw(n) # shear velocity
    # coefficient times wind velocity squared
    KE = shear *  tau * dt # kinetic energy as function of wind
    
    if ice:
      KE = KE * KEice
    
    maxdep = 0
    for dep in range(0, nx-1):
      if dep == 0:
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
             # abs(calc_dens(u[dep+1])- calc_dens(u[dep])))
             abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0])))))
      else:
        PEprior = deepcopy(PE)
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
            # abs(calc_dens(u[dep+1])- calc_dens(u[dep]))) + PEprior
            abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0:(dep+1)])))) + PEprior)
            
      if PE > KE:
        maxdep = dep - 1
        break
      elif dep > 0 and PE < KE:
          u[(dep - 1):(dep+1)] = np.sum(u[(dep-1):(dep+1)] * volume[(dep-1):(dep+1)])/np.sum(volume[(dep-1):(dep+1)])
      
      maxdep = dep
      
    mix[0,idn] = KE/PE #append(mix, KE/PE)
    therm_z[0,idn] = depth[maxdep] #append(therm.z, maxdep)
    
    if pgdl_mode == 'on':
      um_mix[:, idn] = u

    end_time = datetime.datetime.now()
    print("mixing: " + str(end_time - start_time))

    ## (4) DENSITY INSTABILITIES
    # convective overturn: Convective mixing is induced by an unstable density 
    # profile. All groups of water layers where the vertical density profile is 
    # unstable are mixed with the first stable layer below the unstable layer(s) 
    # (i.e., a layer volume weighed means of temperature and other variables are 
    # calculated for the mixed water column). This procedure is continued until 
    # the vertical density profile in the whole water column becomes neutral or stable.
    start_time = datetime.datetime.now()
    dens_u = calc_dens(u) 
    diff_dens_u = np.diff(dens_u) 
    diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
    while np.any(diff_dens_u < 0):
      dens_u = calc_dens(u)
      for dep in range(0, nx-1):
        if dens_u[dep+1] < dens_u[dep] and abs(dens_u[dep+1] - dens_u[dep]) >= denThresh:
          u[(dep):(dep+2)] = np.sum(u[(dep):(dep+2)] * volume[(dep):(dep+2)])/np.sum(volume[(dep):(dep+2)])
          break
        
      dens_u = calc_dens(u)
      diff_dens_u = np.diff(dens_u)
      diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
      
    dens_u_n2 = calc_dens(u)
    n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
    if np.max(n2) > 1e-4:
      max_n2 = depth[np.argmax(n2)]
    else:
      max_n2 = np.max(depth)
    mix_z[0, idn] = max_n2
    if pgdl_mode == 'on':
      um_conv[:, idn] = u
      
    
    end_time = datetime.datetime.now()
    print("convection: " + str(end_time - start_time))
    ## (5) ICE FORMATION
    # according to Saloranta & Andersen (2007) and ice growth due to Stefan's law
    # (Leppäranta 1991)
    start_time = datetime.datetime.now()
    icep  = max(dt_iceon_avg,  (dt/86400))
    x = (dt/86400) / icep
    iceT = iceT * (1 - x) + u[0] * x
    
    K_snow = 2.22362 * (rho_snow/1000)**1.885
    Tice = 0
    
    
    if (iceT <= 0) and Hi < Ice_min and Tair(n) <= 0 and ice == False:
      supercooled = u < 0
      initEnergy = np.sum((0-u[supercooled])*area[supercooled] * dx * Cw)
      
      Hi = Ice_min+(initEnergy/(910*L_ice))/np.max(area)
      
      ice = True
      
      if Hi >= 0:
          
        u[supercooled] = 0
        u[0] = 0
        
    elif ice == True and Hi >= Ice_min:
        Q_surf = (u[0] - 0) * Cw * dx
        u[0] = 0
        
        if Tair(n) > 0:
            Tice = 0
            dHsnew = 0
            
            if (Hs > 0):
                dHs = (-1) * np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice)])
                if (Hs + dHs) < 0:
                    Hi_new = Hi + (Hs + dHs) * (rho_fw/rho_ice)
                else:
                    Hi_new = Hi
            else:
                dHs = 0
                
                Hi_new = Hi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                Hsi = Hsi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                if Hsi <= 0:
                    Hsi = 0
        else:
            if Hs > 0:
                K_snow = 2.22362 * (rho_snow/1000)**(1.885)
                p = (K_ice/K_snow) * (((rho_fw/rho_snow) * Hs ) / Hi)
                dHsi = np.max([0, Hi * (rho_ice/rho_fw -1) + Hs])
                Hsi = Hsi + dHsi

            else:
                p = 1/(10 * Hi)
                dHsi = 0
            
            Tice = (p * 0 + Tair(n)) / (1 + p)
            Hi_new = np.sqrt((Hi + dHsi)**2 + 2 * K_ice/(rho_ice * L_ice)* (0 - Tice) * dt)
            
            # PRECIPITATION
            dHsnew = PP(n) * 1/(1000 * 86400) * dt

            dHs = dHsnew - dHsi * (rho_ice/rho_fw)
            dHsi = 0   

                
        Hi = Hi_new - np.max([0,(Q_surf/(rho_ice * L_ice))])
        
        # if Hi > 2:
        #     print(Hi)
        #     print(meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
        #                                                                            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
        #                                                                            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
        #                                                                            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice))
        #     breakpoint()
    
        Q_surf = 0

        Hs = Hs + dHs
        
        

    
    
        if Hi < Hsi:
            Hsi = np.max([0, Hi])
        
        if Hs <= 0:
            Hs = 0
            rho_snow = rho_new_snow
        else:
            rho_snow = rho_snow * (Hs - dHsnew) / Hs + rho_new_snow * dHsnew/Hs
    elif ice and Hi <= Ice_min:
        ice = False
    
    if (ice == False):
        Hi = 0
        Hs = 0
        Hsi = 0
    
    
    Him[0,idn] = Hi
    Hsm[0,idn] = Hs
    Hsim[0,idn] = Hsi
    Ticem[0,idn] = Tice
    

    last_ice = Hi
    last_snow = Hs
    last_snowice = Hsi

    
    n2m[:,idn] = n2
    um[:,idn] = u
    
    if pgdl_mode == 'on':
      um_ice[:, idn] = u
      meteo_pgdl[0, idn] = Tair(n)
      meteo_pgdl[1, idn] = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) -
        backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps))
      meteo_pgdl[2, idn] = latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[3, idn] = sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[4, idn] = Jsw(n)
      meteo_pgdl[5, idn] = kd_light
      meteo_pgdl[6, idn] = shear
      meteo_pgdl[7, idn] = tau
      meteo_pgdl[8, idn] = np.nanmax(area)
      
    end_time = datetime.datetime.now()
    print("ice: " + str(end_time - start_time))

  
  bf_sim = np.apply_along_axis(center_buoyancy, axis=1, arr = um.T, depths=depth)
  

  df_z_df_sim = pd.DataFrame({'time': times, 'thermoclineDep': bf_sim})

  df_z_df_sim['epi'] = np.nan
  df_z_df_sim['hypo'] = np.nan
  df_z_df_sim['tot'] = np.nan
  df_z_df_sim['stratFlag'] = np.nan
  for j in range(df_z_df_sim.shape[0]):
    if np.isnan(df_z_df_sim.loc[j, 'thermoclineDep']):
      cur_z = 1
      cur_ind = 0
    else:
      cur_z = df_z_df_sim.loc[j, 'thermoclineDep']
      cur_ind = np.max(np.where(depth < cur_z))
      
    df_z_df_sim.loc[j, 'epi'] = np.sum(um[0:(cur_ind + 1), j] * area[0:(cur_ind+1)]) / np.sum(area[0:(cur_ind+1)])
    df_z_df_sim.loc[j, 'hypo'] = np.sum(um[ cur_ind:, j] * area[cur_ind:]) / np.sum(area[cur_ind:])
    df_z_df_sim.loc[j, 'tot'] = np.sum(um[:,j] * area) / np.sum(area)
    if calc_dens(um[-1,j]) - calc_dens(um[0,j]) >= 0.1 and np.mean(um[:,j]) >= 4:
      df_z_df_sim.loc[j, 'stratFlag'] = 1
    else:
      df_z_df_sim.loc[j, 'stratFlag'] = 0
  #breakpoint()
  dat = {'temp' : um,
          'diff' : kzm,
          'mixing' : mix,
          'buoyancy' : n2m,
          'icethickness' : Hi,
          'iceflag' : ice,
          'icemovAvg' : iceT,
          'supercooled' : supercooled,
          'mixingdepth' : mix_z,
          'thermoclinedepth' : therm_z,
          'endtime' : endTime, 
          'average' : df_z_df_sim,
          'last_ice' : last_ice,
          'last_snow' : last_snow,
          'last_snowice' : last_snowice,
          'density_snow' : rho_snow}
  if pgdl_mode == 'on':
    dat = {'temp' : um,
               'diff' : kzm,
               'mixing' : mix,
               'buoyancy' : n2m,
               'icethickness' : Him,
               'snowthickness' : Hsm,
               'snowicethickness' : Hsim,
               'iceflag' : ice,
               'icemovAvg' : iceT,
               'supercooled' : supercooled,
               'mixingdepth' : mix_z,
               'thermoclinedepth' : therm_z,
               'endtime' : endTime, 
               'average' : df_z_df_sim,
               'temp_initial' : um_initial,
               'temp_heat' : um_heat,
               'temp_diff' : um_diff,
               'temp_mix' : um_mix,
               'temp_conv' : um_conv,
               'temp_ice' : um_ice,
               'meteo_input' : meteo_pgdl,
               'buoyancy_pgdl' : n2_pgdl,
               'heatflux_lwsl' : Qm,
               'heatflux_sw' : Hm,
               'last_ice' : last_ice,
               'last_snow' : last_snow,
               'last_snowice' : last_snowice,
               'density_snow' : rho_snow}
  
  return(dat)

def run_thermalmodel_v2(
  u, 
  startTime, 
  endTime,
  area,
  volume,
  depth,
  zmax,
  nx,
  dt,
  dx,
  daily_meteo,
  secview,
  ice=False,
  Hi=0,
  iceT=6,
  supercooled=0,
  diffusion_method = 'hendersonSellers',
  scheme='implicit',
  mixing = 0,
  kd_light=None,
  denThresh=1e-3,
  albedo=0.1,
  eps=0.97,
  emissivity=0.97,
  sigma=5.67e-8,
  sw_factor = 1.0,
  wind_factor = 1.0,
  p2=1,
  B=0.61,
  g=9.81,
  Cd=0.0013, # momentum coeff (wind)
  meltP=1,
  dt_iceon_avg=0.8,
  Hgeo=0.1, # geothermal heat
  KEice=1/1000,
  Ice_min=0.1,
  pgdl_mode='on',
  Hs = 0,
  rho_snow = 250,
  Hsi = 0,
  rho_ice = 910,
  rho_fw = 1000,
  rho_new_snow = 250,
  rho_max_snow = 450,
  K_ice = 2.1,
  Cw = 4.18E6,
  L_ice = 333500,
  kd_snow = 0.9,
  kd_ice = 0.7):
    

  
  ## linearization of driver data, so model can have dynamic step
  Jsw_fillvals = tuple(daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values[[0, -1]])
  Jsw = interp1d(daily_meteo.dt.values, daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jsw_fillvals, bounds_error=False)
  Jlw_fillvals = tuple(daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values[[0,-1]])
  Jlw = interp1d(daily_meteo.dt.values, daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jlw_fillvals, bounds_error=False)
  Tair_fillvals = tuple(daily_meteo.Air_Temperature_celsius.values[[0,-1]])
  Tair = interp1d(daily_meteo.dt.values, daily_meteo.Air_Temperature_celsius.values, kind = "linear", fill_value=Tair_fillvals, bounds_error=False)
  ea_fillvals = tuple(daily_meteo.ea.values[[0,-1]])
  ea = interp1d(daily_meteo.dt.values, daily_meteo.ea.values, kind = "linear", fill_value=ea_fillvals, bounds_error=False)
  Uw_fillvals = tuple(daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values[[0, -1]])
  Uw = interp1d(daily_meteo.dt.values, wind_factor * daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values, kind = "linear", fill_value=Uw_fillvals, bounds_error=False)
  CC_fillvals = tuple(daily_meteo.Cloud_Cover.values[[0,-1]])
  CC = interp1d(daily_meteo.dt.values, daily_meteo.Cloud_Cover.values, kind = "linear", fill_value=CC_fillvals, bounds_error=False)
  Pa_fillvals = tuple(daily_meteo.Surface_Level_Barometric_Pressure_pascal.values[[0,-1]])
  Pa = interp1d(daily_meteo.dt.values, daily_meteo.Surface_Level_Barometric_Pressure_pascal.values, kind = "linear", fill_value=Pa_fillvals, bounds_error=False)
  if kd_light is None:
      kd_fillvals = tuple(secview.kd.values[[0,-1]])
      kd = interp1d(secview.dt.values, secview.kd.values, kind = "linear", fill_value=kd_fillvals, bounds_error=False)
  RH_fillvals = tuple(daily_meteo.Relative_Humidity_percent.values[[0,-1]])
  RH = interp1d(daily_meteo.dt.values, daily_meteo.Relative_Humidity_percent.values, kind = "linear", fill_value=RH_fillvals, bounds_error=False)
  PP_fillvals = tuple(daily_meteo.Precipitation_millimeterPerDay.values[[0,-1]])
  PP = interp1d(daily_meteo.dt.values, daily_meteo.Precipitation_millimeterPerDay.values, kind = "linear", fill_value=PP_fillvals, bounds_error=False)

  #plt.plot(PP(np.arange(1,1e7,1)))
  #plt.plot(daily_meteo.Precipitation_millimeterPerDay.values[[0, -1]])
  
  step_times = np.arange(startTime, endTime, dt)
  nCol = len(step_times)
  um = np.full([nx, nCol], np.nan)
  kzm = np.full([nx, nCol], np.nan)
  n2m = np.full([(nx-1), nCol], np.nan)
  mix = np.full([1,nCol], np.nan)
  therm_z = np.full([1,nCol], np.nan)
  mix_z = np.full([1,nCol], np.nan)
  Him = np.full([1,nCol], np.nan)
  Hm = np.full([nx, nCol], np.nan) 
  Qm = np.full([1,nCol], np.nan)
  Him= np.full([1,nCol], np.nan)
  Hsm= np.full([1,nCol], np.nan)
  Hsim= np.full([1,nCol], np.nan)
  Ticem= np.full([1,nCol], np.nan)
  Similarity = np.full([5,nCol], np.nan)
  
  if pgdl_mode == 'on':
    um_initial = np.full([nx, nCol], np.nan)
    um_heat = np.full([nx, nCol], np.nan)
    um_diff = np.full([nx, nCol], np.nan)
    um_mix = np.full([nx, nCol], np.nan)
    um_conv = np.full([nx, nCol], np.nan)
    um_ice = np.full([nx, nCol], np.nan)
    n2_pgdl = np.full([nx, nCol], np.nan)
    meteo_pgdl = np.full([9, nCol], np.nan)
  
  if not kd_light is None:
    def kd(n): # using this shortcut for now / testing if it works
      return kd_light

  

  times = np.arange(startTime, endTime, dt)
  for idn, n in enumerate(times):
    
    un = deepcopy(u)
    un_initial = un
    dens_u_n2 = calc_dens(u)
    time_ind = np.where(times == n)
    
    if pgdl_mode == 'on':
      n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
      n2_pgdl[:,idn] = np.concatenate([n2, np.array([np.nan])])
      um_initial[:, idn] = u
      
      
    wind_mixing = Uw(n)
    
    if 'kzn' in locals():
        1+1
    else: 
        kzn = u * 0.0
    
    
    ## (2) DIFFUSION
    if diffusion_method == 'hendersonSellers':
        kz = eddy_diffusivity_hendersonSellers(dens_u_n2, depth, 9.81, np.mean(dens_u_n2) , ice, area, wind_mixing,  43.100948, u, kzn) / 1
    elif diffusion_method == 'munkAnderson':
        kz = eddy_diffusivity_munkAnderson(dens_u_n2, depth, 9.81, np.mean(dens_u_n2) , ice, area, wind_mixing,  43.100948, Cd, u, kzn) / 1
    elif diffusion_method == 'hondzoStefan':
        kz = eddy_diffusivity(dens_u_n2, depth, 9.81, np.mean(dens_u_n2) , ice, area, u, kzn) / 86400
        
    
    

    if ice and Tair(n) <= 0:
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif (ice and Tair(n) >= 0):
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif not ice:
      albedo = 0.1
      IceSnowAttCoeff = 1
    
    
    ## (1) HEAT ADDITION
    # surface heat flux
    start_time = datetime.datetime.now()
    Q = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd))  
    
    # heat addition over depth
    
    
    if ice:
        H =  IceSnowAttCoeff * (Jsw(n) * sw_factor)  * np.exp(-(kd_light) * depth)
    else:
        H =  (1- albedo) * (Jsw(n) * sw_factor)  * np.exp(-(kd_light ) * depth)
    
    Hg = (area[:-1]-area[1:])/dx * Hgeo/(4181 * calc_dens(un[0]))
    Hg = np.append(Hg, Hg.min())
    
    u[0] = (un[0] + 
        (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
        Hg[0]) * dt/area[0])
      # all layers in between
    for i in range(1,(nx-1)):
         u[i] = un[i] + (abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i]
      # bottom layer
    u[(nx-1)] = un[(nx-1)] + (abs(H[(nx-1)]-H[(nx-2)]) * area[(nx-1)]/(area[(nx-1)] * dx) * 1/(4181 * calc_dens(un[(nx-1)])) +Hg[(nx-1)]/area[(nx-1)]) * dt

    if pgdl_mode == 'on':
      um_heat[:, idn] = u
      Hm[:, idn] = H
      Qm[0, idn] = Q
    
    un = u
    Similarity[0, idn] = pearsonr(un_initial, u)[0]
    
    end_time = datetime.datetime.now()
    print("heating: " + str(end_time - start_time))
    

    
    kzn = kz
    kzm[:,idn] = kzn
    
    start_time = datetime.datetime.now()
    if scheme == 'implicit':

      
        # IMPLEMENTATION OF CRANK-NICHOLSON SCHEME

        j = len(un)
        y = np.zeros((len(un), len(un)))
        
        # alpha = (dt/dx**2) * kzn 
        # alpha = (dt/dx) * kzn 
        alpha = (area * kzn * dt) / (2 * dx**2)
        beta = (2 * dx**2) / (kzn * dt)
        
        az = - alpha # subdiagonal
        # bz = 2 * (1 + alpha) # diagonal
        bz = (area + 2 * alpha) # diagonal
        cz = - alpha # superdiagonal
        
        bz[0] = 1
        # az[len(az)-2] = 0
        bz[len(bz)-1] = 1
        cz[0] = 0
        
        az =  np.delete(az,0)
        cz =  np.delete(cz,len(cz)-1)
        
        # tridiagonal matrix
        for k in range(j-1):
            y[k][k] = bz[k]
            y[k][k+1] = cz[k]
            y[k+1][k] = az[k]
        
        # y[0,1] = 0    
        # y[j-1, j-1] = 1
        y[j-1, j-2] = 0
        y[j-1, j-1] = 1

        # print(y[0:4])
        
        mn = un * 0.0    
        mn[0] = u[0]
        mn[len(mn)-1] = u[len(u)-1]
        
        for k in range(1,j-1):
            # mn[k] = alpha[k] * u[k-1] + 2 * (1 - alpha[k]) * u[k] + alpha[k] * u[k+1]
            mn[k] = alpha[k] * u[k-1] + (area[k] - 2 * alpha[k]) * u[k] + alpha[k] * u[k+1]

    # DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE
    
        #breakpoint()

#        breakpoint()
        un = u
        u = np.linalg.solve(y, mn)
        
        Similarity[1, idn] = pearsonr(un_initial, u)[0]
        
        #u = crank_nicholson(T = un, dz = dx, dt = dt, kappa = kzn, depth  = depth, area = area, volume = volume)

        
    # TODO: implement / figure out this
    if scheme == 'explicit':
     
      u[0]= un[0]
      u[-1] = un[-1]
      for i in range(1,(nx-1)):
        u[i] = (un[i] + (kzn[i] * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])))
      
        # u[0] = (un[0] + 
      #   (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
      #   Hg[0]) * dt/area[0])
      # all layers in between
      # for i in range(1,(nx-1)):
      #   u[i] = (un[i] + (area[i] * kzn[i] * 1 / dx**2 * (un[i+1] - 2 * un[i] + un[i-1]) +
      #     abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i])
      # bottom layer
      # u[(nx-1)] = (un[(nx-1)] +
      # (abs(H[(nx-1)]-H[(nx-1)-1]) * area[(nx-1)]/(area[(nx-1)]*dx) * 1/(4181 * calc_dens(un[(nx-1)])) +
      # Hg[(nx-1)]/area[(nx-1)]) * dt)
      
                                                           
    if pgdl_mode == 'on':
      um_diff[:, idn] = u

    end_time = datetime.datetime.now()
    print("diffusion: " + str(end_time - start_time))
      
    ## (3) TURBULENT MIXING OF MIXED LAYER
    # the mixed layer depth is determined for each time step by comparing kinetic 
    # energy available from wind and the potential energy required to completely 
    # mix the water column to a given depth
    start_time = datetime.datetime.now()
    Zcv = np.sum(depth * area) / sum(area)  # center of volume
    tau = 1.225 * Cd * Uw(n) ** 2 # wind shear is air density times wind velocity 
    if (Uw(n) <= 15):
      c10 = 0.0005 * sqrt(Uw(n))
    else:
      c10 = 0.0026
    
    un = u
    shear = sqrt((c10 * calc_dens(un[0]))/1.225) *  Uw(n) # shear velocity
    # coefficient times wind velocity squared
    KE = shear *  tau * dt # kinetic energy as function of wind
    
    if ice:
      KE = KE * KEice
    
    maxdep = 0
    for dep in range(0, nx-1):
      if dep == 0:
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
             # abs(calc_dens(u[dep+1])- calc_dens(u[dep])))
             abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0])))))
      else:
        PEprior = deepcopy(PE)
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
            # abs(calc_dens(u[dep+1])- calc_dens(u[dep]))) + PEprior
            abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0:(dep+1)])))) + PEprior)
            
      if PE > KE:
        maxdep = dep - 1
        break
      elif dep > 0 and PE < KE:
          if mixing == 1:
              u[(dep - 1):(dep+1)] = np.sum(u[(dep-1):(dep+1)] * volume[(dep-1):(dep+1)])/np.sum(volume[(dep-1):(dep+1)])
      
      maxdep = dep
      
    mix[0,idn] = KE/PE #append(mix, KE/PE)
    therm_z[0,idn] = depth[maxdep] #append(therm.z, maxdep)
    
    if pgdl_mode == 'on':
      um_mix[:, idn] = u

    Similarity[2, idn] = pearsonr(un_initial, u)[0]
    end_time = datetime.datetime.now()
    print("mixing: " + str(end_time - start_time))

    ## (4) DENSITY INSTABILITIES
    # convective overturn: Convective mixing is induced by an unstable density 
    # profile. All groups of water layers where the vertical density profile is 
    # unstable are mixed with the first stable layer below the unstable layer(s) 
    # (i.e., a layer volume weighed means of temperature and other variables are 
    # calculated for the mixed water column). This procedure is continued until 
    # the vertical density profile in the whole water column becomes neutral or stable.
    start_time = datetime.datetime.now()
    dens_u = calc_dens(u) 
    diff_dens_u = np.diff(dens_u) 
    diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
    un = u 
    while np.any(diff_dens_u < 0):
      dens_u = calc_dens(u)
      for dep in range(0, nx-1):
        if dens_u[dep+1] < dens_u[dep] and abs(dens_u[dep+1] - dens_u[dep]) >= denThresh:
          u[(dep):(dep+2)] = np.sum(u[(dep):(dep+2)] * volume[(dep):(dep+2)])/np.sum(volume[(dep):(dep+2)])
          break
        
      dens_u = calc_dens(u)
      diff_dens_u = np.diff(dens_u)
      diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
      
    dens_u_n2 = calc_dens(u)
    n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
    if np.max(n2) > 1e-4:
      max_n2 = depth[np.argmax(n2)]
    else:
      max_n2 = np.max(depth)
    mix_z[0, idn] = max_n2
    if pgdl_mode == 'on':
      um_conv[:, idn] = u
    Similarity[3, idn] = pearsonr(un_initial, u)[0]
    
    end_time = datetime.datetime.now()
    print("convection: " + str(end_time - start_time))
    ## (5) ICE FORMATION
    # according to Saloranta & Andersen (2007) and ice growth due to Stefan's law
    # (Leppäranta 1991)
    start_time = datetime.datetime.now()
    icep  = max(dt_iceon_avg,  (dt/86400))
    x = (dt/86400) / icep
    iceT = iceT * (1 - x) + u[0] * x
    un = u
    
    K_snow = 2.22362 * (rho_snow/1000)**1.885
    Tice = 0
    
    
    if (iceT <= 0) and Hi < Ice_min and Tair(n) <= 0 and ice == False:
      supercooled = u < 0
      initEnergy = np.sum((0-u[supercooled])*area[supercooled] * dx * Cw)
      
      Hi = Ice_min+(initEnergy/(910*L_ice))/np.max(area)
      
      ice = True
      
      if Hi >= 0:
          
        u[supercooled] = 0
        u[0] = 0
        
    elif ice == True and Hi >= Ice_min:
        Q_surf = (u[0] - 0) * Cw * dx
        u[0] = 0
        
        if Tair(n) > 0:
            Tice = 0
            dHsnew = 0
            
            if (Hs > 0):
                dHs = (-1) * np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice)])
                if (Hs + dHs) < 0:
                    Hi_new = Hi + (Hs + dHs) * (rho_fw/rho_ice)
                else:
                    Hi_new = Hi
            else:
                dHs = 0
                
                Hi_new = Hi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                Hsi = Hsi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                if Hsi <= 0:
                    Hsi = 0
        else:
            if Hs > 0:
                K_snow = 2.22362 * (rho_snow/1000)**(1.885)
                p = (K_ice/K_snow) * (((rho_fw/rho_snow) * Hs ) / Hi)
                dHsi = np.max([0, Hi * (rho_ice/rho_fw -1) + Hs])
                Hsi = Hsi + dHsi

            else:
                p = 1/(10 * Hi)
                dHsi = 0
            
            Tice = (p * 0 + Tair(n)) / (1 + p)
            Hi_new = np.sqrt((Hi + dHsi)**2 + 2 * K_ice/(rho_ice * L_ice)* (0 - Tice) * dt)
            
            # PRECIPITATION
            dHsnew = PP(n) * 1/(1000 * 86400) * dt

            dHs = dHsnew - dHsi * (rho_ice/rho_fw)
            dHsi = 0   

                
        Hi = Hi_new - np.max([0,(Q_surf/(rho_ice * L_ice))])
        
        # if Hi > 2:
        #     print(Hi)
        #     print(meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
        #                                                                            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
        #                                                                            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
        #                                                                            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice))
        #     breakpoint()
    
        Q_surf = 0

        Hs = Hs + dHs
        
        

    
    
        if Hi < Hsi:
            Hsi = np.max([0, Hi])
        
        if Hs <= 0:
            Hs = 0
            rho_snow = rho_new_snow
        else:
            rho_snow = rho_snow * (Hs - dHsnew) / Hs + rho_new_snow * dHsnew/Hs
    elif ice and Hi <= Ice_min:
        ice = False
    
    if (ice == False):
        Hi = 0
        Hs = 0
        Hsi = 0
    
    
    Similarity[4, idn] = pearsonr(un_initial, u)[0]
    Him[0,idn] = Hi
    Hsm[0,idn] = Hs
    Hsim[0,idn] = Hsi
    Ticem[0,idn] = Tice
    

    last_ice = Hi
    last_snow = Hs
    last_snowice = Hsi

    
    n2m[:,idn] = n2
    um[:,idn] = u
    
    # print( um_heat[:, idn] )
    # print( um_diff[:, idn] )
    # print( um_mix[:, idn] )
    # print( um_conv[:, idn] )
    # print( um[:, idn] )
    # breakpoint()
    
    if pgdl_mode == 'on':
      um_ice[:, idn] = u
      meteo_pgdl[0, idn] = Tair(n)
      meteo_pgdl[1, idn] = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) -
        backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps))
      meteo_pgdl[2, idn] = latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[3, idn] = sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[4, idn] = Jsw(n)
      meteo_pgdl[5, idn] = kd_light
      meteo_pgdl[6, idn] = shear
      meteo_pgdl[7, idn] = tau
      meteo_pgdl[8, idn] = np.nanmax(area)
      
    end_time = datetime.datetime.now()
    print("ice: " + str(end_time - start_time))

  
  bf_sim = np.apply_along_axis(center_buoyancy, axis=1, arr = um.T, depths=depth)
  

  df_z_df_sim = pd.DataFrame({'time': times, 'thermoclineDep': bf_sim})

  df_z_df_sim['epi'] = np.nan
  df_z_df_sim['hypo'] = np.nan
  df_z_df_sim['tot'] = np.nan
  df_z_df_sim['stratFlag'] = np.nan
  for j in range(df_z_df_sim.shape[0]):
    if np.isnan(df_z_df_sim.loc[j, 'thermoclineDep']):
      cur_z = 1
      cur_ind = 0
    else:
      cur_z = df_z_df_sim.loc[j, 'thermoclineDep']
      cur_ind = np.max(np.where(depth < cur_z))
      
    df_z_df_sim.loc[j, 'epi'] = np.sum(um[0:(cur_ind + 1), j] * area[0:(cur_ind+1)]) / np.sum(area[0:(cur_ind+1)])
    df_z_df_sim.loc[j, 'hypo'] = np.sum(um[ cur_ind:, j] * area[cur_ind:]) / np.sum(area[cur_ind:])
    df_z_df_sim.loc[j, 'tot'] = np.sum(um[:,j] * area) / np.sum(area)
    if calc_dens(um[-1,j]) - calc_dens(um[0,j]) >= 0.1 and np.mean(um[:,j]) >= 4:
      df_z_df_sim.loc[j, 'stratFlag'] = 1
    else:
      df_z_df_sim.loc[j, 'stratFlag'] = 0
  #breakpoint()
  dat = {'temp' : um,
          'diff' : kzm,
          'mixing' : mix,
          'buoyancy' : n2m,
          'icethickness' : Hi,
          'iceflag' : ice,
          'icemovAvg' : iceT,
          'supercooled' : supercooled,
          'mixingdepth' : mix_z,
          'thermoclinedepth' : therm_z,
          'endtime' : endTime, 
          'average' : df_z_df_sim,
          'last_ice' : last_ice,
          'last_snow' : last_snow,
          'last_snowice' : last_snowice,
          'density_snow' : rho_snow,
          'similarity' : Similarity}
  if pgdl_mode == 'on':
    dat = {'temp' : um,
               'diff' : kzm,
               'mixing' : mix,
               'buoyancy' : n2m,
               'icethickness' : Him,
               'snowthickness' : Hsm,
               'snowicethickness' : Hsim,
               'iceflag' : ice,
               'icemovAvg' : iceT,
               'supercooled' : supercooled,
               'mixingdepth' : mix_z,
               'thermoclinedepth' : therm_z,
               'endtime' : endTime, 
               'average' : df_z_df_sim,
               'temp_initial' : um_initial,
               'temp_heat' : um_heat,
               'temp_diff' : um_diff,
               'temp_mix' : um_mix,
               'temp_conv' : um_conv,
               'temp_ice' : um_ice,
               'meteo_input' : meteo_pgdl,
               'buoyancy_pgdl' : n2_pgdl,
               'heatflux_lwsl' : Qm,
               'heatflux_sw' : Hm,
               'last_ice' : last_ice,
               'last_snow' : last_snow,
               'last_snowice' : last_snowice,
               'density_snow' : rho_snow,
               'similarity' : Similarity}
  
  return(dat)

def run_hybridmodel_heating(
  u, 
  startTime, 
  endTime,
  area,
  volume,
  depth,
  zmax,
  nx,
  dt,
  dx,
  daily_meteo,
  secview,
  std_scale,
  mean_scale,
  scaler,
  test_input,
  ice=False,
  Hi=0,
  iceT=6,
  supercooled=0,
  scheme='implicit',
  kd_light=None,
  denThresh=1e-3,
  albedo=0.1,
  eps=0.97,
  emissivity=0.97,
  sigma=5.67e-8,
  sw_factor = 1.0,
  p2=1,
  B=0.61,
  g=9.81,
  Cd=0.0013, # momentum coeff (wind)
  meltP=1,
  dt_iceon_avg=0.8,
  Hgeo=0.1, # geothermal heat
  KEice=1/1000,
  Ice_min=0.1,
  pgdl_mode='on',
  Hs = 0,
  rho_snow = 250,
  Hsi = 0,
  rho_ice = 910,
  rho_fw = 1000,
  rho_new_snow = 250,
  rho_max_snow = 450,
  K_ice = 2.1,
  Cw = 4.18E6,
  L_ice = 333500,
  kd_snow = 0.9,
  kd_ice = 0.7):
    
  # dl preamble
  device = torch.device('cpu')
  class MLP(torch.nn.Module):
    def __init__(self, layers, activation="relu", init="xavier"):
        super(MLP, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError("Unspecified activation type")
        
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        if init=="xavier":
            self.xavier_init_weights()
        elif init=="kaiming":
            self.kaiming_init_weights()
    
    def xavier_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Xavier Initialization..")
            for m in self.layers.modules():
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.0)

    def kaiming_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Kaiming Initialization..")
            for m in self.layers.modules():
                if hasattr(m, 'weight'):
                    nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.fill_(0.0)
                        
    def forward(self, x):
        out = self.layers(x)
        return out
    
  class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X
        
    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return len(self.X)
  
  m0_PATH =  f"./../MCL/03_finetuning/saved_models/heating_model_finetuned.pth"
  #m0_PATH =  f"./../MCL/02_training/saved_models/heating_model_time.pth"
  
  m0_layers = [14, 32, 32, 1]

  heating_model = MLP(m0_layers, activation="gelu")
  m0_checkpoint = torch.load(m0_PATH, map_location=torch.device('cpu'))
  heating_model.load_state_dict(m0_checkpoint)
  heating_model = heating_model.to(device)
  
  heating_model.train()
  
  ## linearization of driver data, so model can have dynamic step
  Jsw_fillvals = tuple(daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values[[0, -1]])
  Jsw = interp1d(daily_meteo.dt.values, daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jsw_fillvals, bounds_error=False)
  Jlw_fillvals = tuple(daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values[[0,-1]])
  Jlw = interp1d(daily_meteo.dt.values, daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jlw_fillvals, bounds_error=False)
  Tair_fillvals = tuple(daily_meteo.Air_Temperature_celsius.values[[0,-1]])
  Tair = interp1d(daily_meteo.dt.values, daily_meteo.Air_Temperature_celsius.values, kind = "linear", fill_value=Tair_fillvals, bounds_error=False)
  ea_fillvals = tuple(daily_meteo.ea.values[[0,-1]])
  ea = interp1d(daily_meteo.dt.values, daily_meteo.ea.values, kind = "linear", fill_value=ea_fillvals, bounds_error=False)
  Uw_fillvals = tuple(daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values[[0, -1]])
  Uw = interp1d(daily_meteo.dt.values, daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values, kind = "linear", fill_value=Uw_fillvals, bounds_error=False)
  CC_fillvals = tuple(daily_meteo.Cloud_Cover.values[[0,-1]])
  CC = interp1d(daily_meteo.dt.values, daily_meteo.Cloud_Cover.values, kind = "linear", fill_value=CC_fillvals, bounds_error=False)
  Pa_fillvals = tuple(daily_meteo.Surface_Level_Barometric_Pressure_pascal.values[[0,-1]])
  Pa = interp1d(daily_meteo.dt.values, daily_meteo.Surface_Level_Barometric_Pressure_pascal.values, kind = "linear", fill_value=Pa_fillvals, bounds_error=False)
  if kd_light is None:
      kd_fillvals = tuple(secview.kd.values[[0,-1]])
      kd = interp1d(secview.dt.values, secview.kd.values, kind = "linear", fill_value=kd_fillvals, bounds_error=False)
  RH_fillvals = tuple(daily_meteo.Relative_Humidity_percent.values[[0,-1]])
  RH = interp1d(daily_meteo.dt.values, daily_meteo.Relative_Humidity_percent.values, kind = "linear", fill_value=RH_fillvals, bounds_error=False)
  PP_fillvals = tuple(daily_meteo.Precipitation_millimeterPerDay.values[[0,-1]])
  PP = interp1d(daily_meteo.dt.values, daily_meteo.Precipitation_millimeterPerDay.values, kind = "linear", fill_value=PP_fillvals, bounds_error=False)
  time_fillvals = tuple(daily_meteo.date.values[[0,-1]])
  time = interp1d(daily_meteo.dt.values, daily_meteo.date.values, kind = "linear", fill_value=time_fillvals, bounds_error=False)

  #plt.plot(PP(np.arange(1,1e7,1)))
  #plt.plot(daily_meteo.Precipitation_millimeterPerDay.values[[0, -1]])
  
  step_times = np.arange(startTime, endTime, dt)
  nCol = len(step_times)
  um = np.full([nx, nCol], np.nan)
  kzm = np.full([nx, nCol], np.nan)
  n2m = np.full([(nx-1), nCol], np.nan)
  mix = np.full([1,nCol], np.nan)
  therm_z = np.full([1,nCol], np.nan)
  mix_z = np.full([1,nCol], np.nan)
  Him = np.full([1,nCol], np.nan)
  Hm = np.full([nx, nCol], np.nan) 
  Qm = np.full([1,nCol], np.nan)
  Him= np.full([1,nCol], np.nan)
  Hsm= np.full([1,nCol], np.nan)
  Hsim= np.full([1,nCol], np.nan)
  Ticem= np.full([1,nCol], np.nan)
  
  if pgdl_mode == 'on':
    um_initial = np.full([nx, nCol], np.nan)
    um_heat = np.full([nx, nCol], np.nan)
    um_diff = np.full([nx, nCol], np.nan)
    um_mix = np.full([nx, nCol], np.nan)
    um_conv = np.full([nx, nCol], np.nan)
    um_ice = np.full([nx, nCol], np.nan)
    n2_pgdl = np.full([nx, nCol], np.nan)
    meteo_pgdl = np.full([9, nCol], np.nan)
  
  if not kd_light is None:
    def kd(n): # using this shortcut for now / testing if it works
      return kd_light

  

  times = np.arange(startTime, endTime, dt)
  for idn, n in enumerate(times):
    
    un = deepcopy(u)
    dens_u_n2 = calc_dens(u)
    time_ind = np.where(times == n)
    
    if pgdl_mode == 'on':
      n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
      n2_pgdl[:,idn] = np.concatenate([n2, np.array([np.nan])])
      um_initial[:, idn] = u
      
    kz = eddy_diffusivity(dens_u_n2, depth, 9.81, np.mean(dens_u_n2) , ice, area, u) / 86400
    

    if ice and Tair(n) <= 0:
      kzn = kz
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif (ice and Tair(n) >= 0):
      kzn = kz
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif not ice:
      kzn = kz
      albedo = 0.1
      IceSnowAttCoeff = 1
    kzm[:,idn] = kzn
    
    ## (1) HEAT ADDITION
    start_time = datetime.datetime.now()  

    # surface heat flux
    Q = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd))  
    
    # heat addition over depth
    
    
    if ice:
        H =  IceSnowAttCoeff * (Jsw(n) * sw_factor)  * np.exp(-(kd_light) * depth)
    else:
        H =  (1- albedo) * (Jsw(n) * sw_factor)  * np.exp(-(kd_light ) * depth)
    
    Hg = (area[:-1]-area[1:])/dx * Hgeo/(4181 * calc_dens(un[0]))
    Hg = np.append(Hg, Hg.min())
    
    lngwave_total = longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) -backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps)

    date_time = daily_meteo.date
    day_of_year_list = daily_meteo.day_of_year_list
    time_of_day_list = daily_meteo.time_of_day_list

    input_data_raw = {'depth':[i for i in range(1,26)],
                             'AirTemp_degC':np.ones(25)*Tair(n),
                             'Longwave_Wm-2':np.ones(25)* lngwave_total,
                             'Latent_Wm-2':np.ones(25)* latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd),
                             'Sensible_Wm-2':np.ones(25) * sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd),
                             'Shortwave_Wm-2':np.ones(25) * Jsw(n),
                             'lightExtinct_m-1':np.ones(25) * kd_light,
                             #'ShearVelocity_mS-1':np.ones(25) * shear,
                             #'ShearStress_Nm-2':np.ones(25) * tau,
                             'Area_m2':np.ones(25)* np.nanmax(area),
                             #'buoyancy':np.concatenate([n2, np.array([np.nan])]),
                             'day_of_year':np.ones(25) * day_of_year_list[int(n/dt)],
                             'time_of_day':np.ones(25) * time_of_day_list[int(n/dt)],
                             'ice':np.ones(25) * Hi,
                             'snow':np.ones(25) * Hs,
                             'snowice':np.ones(25) * Hsi,
                             'temp_initial':np.ones(25) * un}
                             #'diffusivity':np.ones(25) * kzn}
    input_mcl = pd.DataFrame(input_data_raw)
    input_columns = ['depth', 'AirTemp_degC', 'Longwave_Wm-2', 'Latent_Wm-2', 'Sensible_Wm-2', 'Shortwave_Wm-2',
                'lightExtinct_m-1', 'Area_m2', 'day_of_year', 'time_of_day', 
                'ice', 'snow', 'snowice', 'temp_initial']
    
    #scaler = StandardScaler()
    #scaler.fit(input_mcl)
    
    input_data = scaler.transform(input_mcl)    

    input_data_tensor = torch.tensor(input_data, device = torch.device('cpu'))
    
    #breakpoint()
    
    output_tensor = heating_model(input_data_tensor.float())
    
    output_array = output_tensor.detach().cpu().numpy()
    

    #u = output_array * input_std[10] + input_mean[10]
    u = output_array * std_scale + mean_scale
    
    # breakpoint()
    
    u = u[:,0]

    end_time = datetime.datetime.now()
    print("heating: " + str(end_time - start_time))
    
    # print(u)
    breakpoint()
    
    # u[0] = (un[0] + 
    #     (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
    #     Hg[0]) * dt/area[0])
    #   # all layers in between
    # for i in range(1,(nx-1)):
    #      u[i] = un[i] + (abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i]
    #   # bottom layer
    # u[(nx-1)] = un[(nx-1)] + (abs(H[(nx-1)]-H[(nx-2)]) * area[(nx-1)]/(area[(nx-1)] * dx) * 1/(4181 * calc_dens(un[(nx-1)])) +Hg[(nx-1)]/area[(nx-1)]) * dt

    if pgdl_mode == 'on':
      um_heat[:, idn] = u
      Hm[:, idn] = H
      Qm[0, idn] = Q


    ## (2) DIFFUSION
    start_time = datetime.datetime.now()  
    if scheme == 'implicit':

      
        # IMPLEMENTATION OF CRANK-NICHOLSON SCHEME
        
        j = len(un)
        y = np.zeros((len(un), len(un)))
        
        alpha = (dt/dx**2) * kzn
        
        az = - alpha # subdiagonal
        bz = 2 * (1 + alpha) # diagonal
        cz = - alpha # superdiagonal
        
        bz[0] = 1
        # az[len(az)-2] = 0
        bz[len(bz)-1] = 1
        cz[0] = 0
        
        az =  np.delete(az,0)
        cz =  np.delete(cz,len(cz)-1)
        
        # tridiagonal matrix
        for k in range(j-1):
            y[k][k] = bz[k]
            y[k][k+1] = cz[k]
            y[k+1][k] = az[k]
        
        # y[0,1] = 0    
        # y[j-1, j-1] = 1
        y[j-1, j-2] = 0
        y[j-1, j-1] = 1
        
        # print(y[0:4])
        
        mn = un * 0.0    
        mn[0] = u[0]
        mn[len(mn)-1] = u[len(u)-1]
        
        for k in range(1,j-2):
            mn[k] = alpha[k] * u[k-1] + 2 * (1 - alpha[k]) * u[k] + alpha[k] * u[k+1]

    # DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE

        u = np.linalg.solve(y, mn)

        
    if scheme == 'explicit':
      u[0] = (un[0] + 
        (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
        Hg[0]) * dt/area[0])
      # all layers in between
      for i in range(1,(nx-1)):
        u[i] = (un[i] + (area[i] * kzn[i] * 1 / dx**2 * (un[i+1] - 2 * un[i] + un[i-1]) +
          abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i])
      # bottom layer
      u[(nx-1)] = (un[(nx-1)] +
      (abs(H[(nx-1)]-H[(nx-1)-1]) * area[(nx-1)]/(area[(nx-1)]*dx) * 1/(4181 * calc_dens(un[(nx-1)])) +
      Hg[(nx-1)]/area[(nx-1)]) * dt)
                                                           
    if pgdl_mode == 'on':
      um_diff[:, idn] = u

    end_time = datetime.datetime.now()
    print("diffusion: " + str(end_time - start_time))
      
    ## (3) TURBULENT MIXING OF MIXED LAYER
    # the mixed layer depth is determined for each time step by comparing kinetic 
    # energy available from wind and the potential energy required to completely 
    # mix the water column to a given depth
    start_time = datetime.datetime.now()  
    Zcv = np.sum(depth * area) / sum(area)  # center of volume
    tau = 1.225 * Cd * Uw(n) ** 2 # wind shear is air density times wind velocity 
    if (Uw(n) <= 15):
      c10 = 0.0005 * sqrt(Uw(n))
    else:
      c10 = 0.0026
      
    shear = sqrt((c10 * calc_dens(un[0]))/1.225) *  Uw(n) # shear velocity
    # coefficient times wind velocity squared
    KE = shear *  tau * dt # kinetic energy as function of wind
    
    if ice:
      KE = KE * KEice
    
    maxdep = 0
    for dep in range(0, nx-1):
      if dep == 0:
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
             # abs(calc_dens(u[dep+1])- calc_dens(u[dep])))
             abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0])))))
      else:
        PEprior = deepcopy(PE)
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
            # abs(calc_dens(u[dep+1])- calc_dens(u[dep]))) + PEprior
            abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0:(dep+1)])))) + PEprior)
            
      if PE > KE:
        maxdep = dep - 1
        break
      elif dep > 0 and PE < KE:
        u[(dep - 1):(dep+1)] = np.sum(u[(dep-1):(dep+1)] * volume[(dep-1):(dep+1)])/np.sum(volume[(dep-1):(dep+1)])
      
      maxdep = dep
      
    mix[0,idn] = KE/PE #append(mix, KE/PE)
    therm_z[0,idn] = depth[maxdep] #append(therm.z, maxdep)
    
    if pgdl_mode == 'on':
      um_mix[:, idn] = u

    end_time = datetime.datetime.now()
    print("mixing: " + str(end_time - start_time))

    ## (4) DENSITY INSTABILITIES
    # convective overturn: Convective mixing is induced by an unstable density 
    # profile. All groups of water layers where the vertical density profile is 
    # unstable are mixed with the first stable layer below the unstable layer(s) 
    # (i.e., a layer volume weighed means of temperature and other variables are 
    # calculated for the mixed water column). This procedure is continued until 
    # the vertical density profile in the whole water column becomes neutral or stable.
    start_time = datetime.datetime.now()  
    dens_u = calc_dens(u) 
    diff_dens_u = np.diff(dens_u) 
    diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
    while np.any(diff_dens_u < 0):
      dens_u = calc_dens(u)
      for dep in range(0, nx-1):
        if dens_u[dep+1] < dens_u[dep] and abs(dens_u[dep+1] - dens_u[dep]) >= denThresh:
          u[(dep):(dep+2)] = np.sum(u[(dep):(dep+2)] * volume[(dep):(dep+2)])/np.sum(volume[(dep):(dep+2)])
          break
        
      dens_u = calc_dens(u)
      diff_dens_u = np.diff(dens_u)
      diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
      
    dens_u_n2 = calc_dens(u)
    n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
    if np.max(n2) > 1e-4:
      max_n2 = depth[np.argmax(n2)]
    else:
      max_n2 = np.max(depth)
    mix_z[0, idn] = max_n2
    if pgdl_mode == 'on':
      um_conv[:, idn] = u
    
    end_time = datetime.datetime.now()
    print("convection: " + str(end_time - start_time))
      
    ## (5) ICE FORMATION
    # according to Saloranta & Andersen (2007) and ice growth due to Stefan's law
    # (Leppäranta 1991)
    start_time = datetime.datetime.now()  
    icep  = max(dt_iceon_avg,  (dt/86400))
    x = (dt/86400) / icep
    iceT = iceT * (1 - x) + u[0] * x
    
    K_snow = 2.22362 * (rho_snow/1000)**1.885
    Tice = 0
    
    
    if (iceT <= 0) and Hi < Ice_min and Tair(n) <= 0 and ice == False:
      supercooled = u < 0
      initEnergy = np.sum((0-u[supercooled])*area[supercooled] * dx * Cw)
      
      Hi = Ice_min+(initEnergy/(910*L_ice))/np.max(area)
      
      ice = True
      
      if Hi >= 0:
          
        u[supercooled] = 0
        u[0] = 0
        
    elif ice == True and Hi >= Ice_min:
        Q_surf = (u[0] - 0) * Cw * dx
        u[0] = 0
        
        if Tair(n) > 0:
            Tice = 0
            dHsnew = 0
            
            if (Hs > 0):
                dHs = (-1) * np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice)])
                if (Hs + dHs) < 0:
                    Hi_new = Hi + (Hs + dHs) * (rho_fw/rho_ice)
                else:
                    Hi_new = Hi
            else:
                dHs = 0
                
                Hi_new = Hi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                Hsi = Hsi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                if Hsi <= 0:
                    Hsi = 0
        else:
            if Hs > 0:
                K_snow = 2.22362 * (rho_snow/1000)**(1.885)
                p = (K_ice/K_snow) * (((rho_fw/rho_snow) * Hs ) / Hi)
                dHsi = np.max([0, Hi * (rho_ice/rho_fw -1) + Hs])
                Hsi = Hsi + dHsi

            else:
                p = 1/(10 * Hi)
                dHsi = 0
            
            Tice = (p * 0 + Tair(n)) / (1 + p)
            Hi_new = np.sqrt((Hi + dHsi)**2 + 2 * K_ice/(rho_ice * L_ice)* (0 - Tice) * dt)
            
            # PRECIPITATION
            dHsnew = PP(n) * 1/(1000 * 86400) * dt

            dHs = dHsnew - dHsi * (rho_ice/rho_fw)
            dHsi = 0   

                
        Hi = Hi_new - np.max([0,(Q_surf/(rho_ice * L_ice))])
        
        # if Hi > 2:
        #     print(Hi)
        #     print(meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
        #                                                                            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
        #                                                                            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
        #                                                                            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice))
        #     breakpoint()
    
        Q_surf = 0

        Hs = Hs + dHs
        
        

    
    
        if Hi < Hsi:
            Hsi = np.max([0, Hi])
        
        if Hs <= 0:
            Hs = 0
            rho_snow = rho_new_snow
        else:
            rho_snow = rho_snow * (Hs - dHsnew) / Hs + rho_new_snow * dHsnew/Hs
    elif ice and Hi <= Ice_min:
        ice = False
    
    if (ice == False):
        Hi = 0
        Hs = 0
        Hsi = 0
    
    
    Him[0,idn] = Hi
    Hsm[0,idn] = Hs
    Hsim[0,idn] = Hsi
    Ticem[0,idn] = Tice
    

    last_ice = Hi
    last_snow = Hs
    last_snowice = Hsi

    
    n2m[:,idn] = n2
    um[:,idn] = u
    
    if pgdl_mode == 'on':
      um_ice[:, idn] = u
      meteo_pgdl[0, idn] = Tair(n)
      meteo_pgdl[1, idn] = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) -
        backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps))
      meteo_pgdl[2, idn] = latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[3, idn] = sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[4, idn] = Jsw(n)
      meteo_pgdl[5, idn] = kd_light
      meteo_pgdl[6, idn] = shear
      meteo_pgdl[7, idn] = tau
      meteo_pgdl[8, idn] = np.nanmax(area)
    
    end_time = datetime.datetime.now()
    print("ice: " + str(end_time - start_time))


  
  bf_sim = np.apply_along_axis(center_buoyancy, axis=1, arr = um.T, depths=depth)
  

  df_z_df_sim = pd.DataFrame({'time': times, 'thermoclineDep': bf_sim})

  df_z_df_sim['epi'] = np.nan
  df_z_df_sim['hypo'] = np.nan
  df_z_df_sim['tot'] = np.nan
  df_z_df_sim['stratFlag'] = np.nan
  for j in range(df_z_df_sim.shape[0]):
    if np.isnan(df_z_df_sim.loc[j, 'thermoclineDep']):
      cur_z = 1
      cur_ind = 0
    else:
      cur_z = df_z_df_sim.loc[j, 'thermoclineDep']
      cur_ind = np.max(np.where(depth < cur_z))
      
    df_z_df_sim.loc[j, 'epi'] = np.sum(um[0:(cur_ind + 1), j] * area[0:(cur_ind+1)]) / np.sum(area[0:(cur_ind+1)])
    df_z_df_sim.loc[j, 'hypo'] = np.sum(um[ cur_ind:, j] * area[cur_ind:]) / np.sum(area[cur_ind:])
    df_z_df_sim.loc[j, 'tot'] = np.sum(um[:,j] * area) / np.sum(area)
    if calc_dens(um[-1,j]) - calc_dens(um[0,j]) >= 0.1 and np.mean(um[:,j]) >= 4:
      df_z_df_sim.loc[j, 'stratFlag'] = 1
    else:
      df_z_df_sim.loc[j, 'stratFlag'] = 0
  #breakpoint()
  dat = {'temp' : um,
          'diff' : kzm,
          'mixing' : mix,
          'buoyancy' : n2m,
          'icethickness' : Hi,
          'iceflag' : ice,
          'icemovAvg' : iceT,
          'supercooled' : supercooled,
          'mixingdepth' : mix_z,
          'thermoclinedepth' : therm_z,
          'endtime' : endTime, 
          'average' : df_z_df_sim,
          'last_ice' : last_ice,
          'last_snow' : last_snow,
          'last_snowice' : last_snowice,
          'density_snow' : rho_snow}
  if pgdl_mode == 'on':
    dat = {'temp' : um,
               'diff' : kzm,
               'mixing' : mix,
               'buoyancy' : n2m,
               'icethickness' : Him,
               'snowthickness' : Hsm,
               'snowicethickness' : Hsim,
               'iceflag' : ice,
               'icemovAvg' : iceT,
               'supercooled' : supercooled,
               'mixingdepth' : mix_z,
               'thermoclinedepth' : therm_z,
               'endtime' : endTime, 
               'average' : df_z_df_sim,
               'temp_initial' : um_initial,
               'temp_heat' : um_heat,
               'temp_diff' : um_diff,
               'temp_mix' : um_mix,
               'temp_conv' : um_conv,
               'temp_ice' : um_ice,
               'meteo_input' : meteo_pgdl,
               'buoyancy_pgdl' : n2_pgdl,
               'heatflux_lwsl' : Qm,
               'heatflux_sw' : Hm,
               'last_ice' : last_ice,
               'last_snow' : last_snow,
               'last_snowice' : last_snowice,
               'density_snow' : rho_snow}
  
  return(dat)


def run_hybridmodel_mixing(
  u, 
  startTime, 
  endTime,
  area,
  volume,
  depth,
  zmax,
  nx,
  dt,
  dx,
  daily_meteo,
  secview,
  std_scale,
  mean_scale,
  scaler,
  test_input,
  ice=False,
  Hi=0,
  iceT=6,
  supercooled=0,
  scheme='implicit',
  kd_light=None,
  denThresh=1e-3,
  albedo=0.1,
  eps=0.97,
  emissivity=0.97,
  sigma=5.67e-8,
  sw_factor = 1.0,
  p2=1,
  B=0.61,
  g=9.81,
  Cd=0.0013, # momentum coeff (wind)
  meltP=1,
  dt_iceon_avg=0.8,
  Hgeo=0.1, # geothermal heat
  KEice=1/1000,
  Ice_min=0.1,
  pgdl_mode='on',
  Hs = 0,
  rho_snow = 250,
  Hsi = 0,
  rho_ice = 910,
  rho_fw = 1000,
  rho_new_snow = 250,
  rho_max_snow = 450,
  K_ice = 2.1,
  Cw = 4.18E6,
  L_ice = 333500,
  kd_snow = 0.9,
  kd_ice = 0.7):
    
  # dl preamble
  device = torch.device('cpu')
  class MLP(torch.nn.Module):
    def __init__(self, layers, activation="relu", init="xavier"):
        super(MLP, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError("Unspecified activation type")
        
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        if init=="xavier":
            self.xavier_init_weights()
        elif init=="kaiming":
            self.kaiming_init_weights()
    
    def xavier_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Xavier Initialization..")
            for m in self.layers.modules():
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.0)

    def kaiming_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Kaiming Initialization..")
            for m in self.layers.modules():
                if hasattr(m, 'weight'):
                    nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.fill_(0.0)
                        
    def forward(self, x):
        out = self.layers(x)
        return out
    
  class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X
        
    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return len(self.X)
  
  m0_PATH =  f"./../MCL/03_finetuning/saved_models/mixing_model_finetuned.pth"
  m0_PATH =  f"./../MCL/02_training/saved_models/mixing_model_time.pth"
  #m0_PATH =  f"./../MCL/02_training/saved_models/heating_model_time.pth"
  
  m0_layers = [9, 32, 32, 1]

  mixing_model = MLP(m0_layers, activation="gelu")
  m0_checkpoint = torch.load(m0_PATH, map_location=torch.device('cpu'))
  mixing_model.load_state_dict(m0_checkpoint)
  mixing_model = mixing_model.to(device)
  
  mixing_model.train()
  
  ## linearization of driver data, so model can have dynamic step
  Jsw_fillvals = tuple(daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values[[0, -1]])
  Jsw = interp1d(daily_meteo.dt.values, daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jsw_fillvals, bounds_error=False)
  Jlw_fillvals = tuple(daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values[[0,-1]])
  Jlw = interp1d(daily_meteo.dt.values, daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jlw_fillvals, bounds_error=False)
  Tair_fillvals = tuple(daily_meteo.Air_Temperature_celsius.values[[0,-1]])
  Tair = interp1d(daily_meteo.dt.values, daily_meteo.Air_Temperature_celsius.values, kind = "linear", fill_value=Tair_fillvals, bounds_error=False)
  ea_fillvals = tuple(daily_meteo.ea.values[[0,-1]])
  ea = interp1d(daily_meteo.dt.values, daily_meteo.ea.values, kind = "linear", fill_value=ea_fillvals, bounds_error=False)
  Uw_fillvals = tuple(daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values[[0, -1]])
  Uw = interp1d(daily_meteo.dt.values, daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values, kind = "linear", fill_value=Uw_fillvals, bounds_error=False)
  CC_fillvals = tuple(daily_meteo.Cloud_Cover.values[[0,-1]])
  CC = interp1d(daily_meteo.dt.values, daily_meteo.Cloud_Cover.values, kind = "linear", fill_value=CC_fillvals, bounds_error=False)
  Pa_fillvals = tuple(daily_meteo.Surface_Level_Barometric_Pressure_pascal.values[[0,-1]])
  Pa = interp1d(daily_meteo.dt.values, daily_meteo.Surface_Level_Barometric_Pressure_pascal.values, kind = "linear", fill_value=Pa_fillvals, bounds_error=False)
  if kd_light is None:
      kd_fillvals = tuple(secview.kd.values[[0,-1]])
      kd = interp1d(secview.dt.values, secview.kd.values, kind = "linear", fill_value=kd_fillvals, bounds_error=False)
  RH_fillvals = tuple(daily_meteo.Relative_Humidity_percent.values[[0,-1]])
  RH = interp1d(daily_meteo.dt.values, daily_meteo.Relative_Humidity_percent.values, kind = "linear", fill_value=RH_fillvals, bounds_error=False)
  PP_fillvals = tuple(daily_meteo.Precipitation_millimeterPerDay.values[[0,-1]])
  PP = interp1d(daily_meteo.dt.values, daily_meteo.Precipitation_millimeterPerDay.values, kind = "linear", fill_value=PP_fillvals, bounds_error=False)
  time_fillvals = tuple(daily_meteo.date.values[[0,-1]])
  time = interp1d(daily_meteo.dt.values, daily_meteo.date.values, kind = "linear", fill_value=time_fillvals, bounds_error=False)

  #plt.plot(PP(np.arange(1,1e7,1)))
  #plt.plot(daily_meteo.Precipitation_millimeterPerDay.values[[0, -1]])
  
  step_times = np.arange(startTime, endTime, dt)
  nCol = len(step_times)
  um = np.full([nx, nCol], np.nan)
  kzm = np.full([nx, nCol], np.nan)
  n2m = np.full([(nx-1), nCol], np.nan)
  mix = np.full([1,nCol], np.nan)
  therm_z = np.full([1,nCol], np.nan)
  mix_z = np.full([1,nCol], np.nan)
  Him = np.full([1,nCol], np.nan)
  Hm = np.full([nx, nCol], np.nan) 
  Qm = np.full([1,nCol], np.nan)
  Him= np.full([1,nCol], np.nan)
  Hsm= np.full([1,nCol], np.nan)
  Hsim= np.full([1,nCol], np.nan)
  Ticem= np.full([1,nCol], np.nan)
  
  if pgdl_mode == 'on':
    um_initial = np.full([nx, nCol], np.nan)
    um_heat = np.full([nx, nCol], np.nan)
    um_diff = np.full([nx, nCol], np.nan)
    um_mix = np.full([nx, nCol], np.nan)
    um_conv = np.full([nx, nCol], np.nan)
    um_ice = np.full([nx, nCol], np.nan)
    n2_pgdl = np.full([nx, nCol], np.nan)
    meteo_pgdl = np.full([9, nCol], np.nan)
  
  if not kd_light is None:
    def kd(n): # using this shortcut for now / testing if it works
      return kd_light

  

  times = np.arange(startTime, endTime, dt)
  for idn, n in enumerate(times):
    
    un = deepcopy(u)
    dens_u_n2 = calc_dens(u)
    time_ind = np.where(times == n)
    
    if pgdl_mode == 'on':
      n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
      n2_pgdl[:,idn] = np.concatenate([n2, np.array([np.nan])])
      um_initial[:, idn] = u
      
    kz = eddy_diffusivity(dens_u_n2, depth, 9.81, np.mean(dens_u_n2) , ice, area, u) / 86400
    

    if ice and Tair(n) <= 0:
      kzn = kz
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif (ice and Tair(n) >= 0):
      kzn = kz
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif not ice:
      kzn = kz
      albedo = 0.1
      IceSnowAttCoeff = 1
    kzm[:,idn] = kzn
    
        ## (1) HEAT ADDITION
    # surface heat flux
    start_time = datetime.datetime.now()
    Q = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd))  
    
    # heat addition over depth
    
    
    if ice:
        H =  IceSnowAttCoeff * (Jsw(n) * sw_factor)  * np.exp(-(kd_light) * depth)
    else:
        H =  (1- albedo) * (Jsw(n) * sw_factor)  * np.exp(-(kd_light ) * depth)
    
    Hg = (area[:-1]-area[1:])/dx * Hgeo/(4181 * calc_dens(un[0]))
    Hg = np.append(Hg, Hg.min())
    
    u[0] = (un[0] + 
        (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
        Hg[0]) * dt/area[0])
      # all layers in between
    for i in range(1,(nx-1)):
         u[i] = un[i] + (abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i]
      # bottom layer
    u[(nx-1)] = un[(nx-1)] + (abs(H[(nx-1)]-H[(nx-2)]) * area[(nx-1)]/(area[(nx-1)] * dx) * 1/(4181 * calc_dens(un[(nx-1)])) +Hg[(nx-1)]/area[(nx-1)]) * dt

    if pgdl_mode == 'on':
      um_heat[:, idn] = u
      Hm[:, idn] = H
      Qm[0, idn] = Q
    
    end_time = datetime.datetime.now()
    print("heating: " + str(end_time - start_time))
    
    
    # u[0] = (un[0] + 
    #     (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
    #     Hg[0]) * dt/area[0])
    #   # all layers in between
    # for i in range(1,(nx-1)):
    #      u[i] = un[i] + (abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i]
    #   # bottom layer
    # u[(nx-1)] = un[(nx-1)] + (abs(H[(nx-1)]-H[(nx-2)]) * area[(nx-1)]/(area[(nx-1)] * dx) * 1/(4181 * calc_dens(un[(nx-1)])) +Hg[(nx-1)]/area[(nx-1)]) * dt

    if pgdl_mode == 'on':
      um_heat[:, idn] = u
      Hm[:, idn] = H
      Qm[0, idn] = Q


    ## (2) DIFFUSION
    start_time = datetime.datetime.now()  
    if scheme == 'implicit':

      
        # IMPLEMENTATION OF CRANK-NICHOLSON SCHEME
        
        j = len(un)
        y = np.zeros((len(un), len(un)))
        
        alpha = (dt/dx**2) * kzn
        
        az = - alpha # subdiagonal
        bz = 2 * (1 + alpha) # diagonal
        cz = - alpha # superdiagonal
        
        bz[0] = 1
        # az[len(az)-2] = 0
        bz[len(bz)-1] = 1
        cz[0] = 0
        
        az =  np.delete(az,0)
        cz =  np.delete(cz,len(cz)-1)
        
        # tridiagonal matrix
        for k in range(j-1):
            y[k][k] = bz[k]
            y[k][k+1] = cz[k]
            y[k+1][k] = az[k]
        
        # y[0,1] = 0    
        # y[j-1, j-1] = 1
        y[j-1, j-2] = 0
        y[j-1, j-1] = 1
        
        # print(y[0:4])
        
        mn = un * 0.0    
        mn[0] = u[0]
        mn[len(mn)-1] = u[len(u)-1]
        
        for k in range(1,j-2):
            mn[k] = alpha[k] * u[k-1] + 2 * (1 - alpha[k]) * u[k] + alpha[k] * u[k+1]

    # DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE

        u = np.linalg.solve(y, mn)

        
    if scheme == 'explicit':
      u[0] = (un[0] + 
        (Q * area[0]/(dx)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + 
        Hg[0]) * dt/area[0])
      # all layers in between
      for i in range(1,(nx-1)):
        u[i] = (un[i] + (area[i] * kzn[i] * 1 / dx**2 * (un[i+1] - 2 * un[i] + un[i-1]) +
          abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i])
      # bottom layer
      u[(nx-1)] = (un[(nx-1)] +
      (abs(H[(nx-1)]-H[(nx-1)-1]) * area[(nx-1)]/(area[(nx-1)]*dx) * 1/(4181 * calc_dens(un[(nx-1)])) +
      Hg[(nx-1)]/area[(nx-1)]) * dt)
                                                           
    if pgdl_mode == 'on':
      um_diff[:, idn] = u

    end_time = datetime.datetime.now()
    print("diffusion: " + str(end_time - start_time))
      
    ## (3) TURBULENT MIXING OF MIXED LAYER
    # the mixed layer depth is determined for each time step by comparing kinetic 
    # energy available from wind and the potential energy required to completely 
    # mix the water column to a given depth
    start_time = datetime.datetime.now()  
    Zcv = np.sum(depth * area) / sum(area)  # center of volume
    tau = 1.225 * Cd * Uw(n) ** 2 # wind shear is air density times wind velocity 
    if (Uw(n) <= 15):
      c10 = 0.0005 * sqrt(Uw(n))
    else:
      c10 = 0.0026
    
    print(u[0])
    print(c10)
    print(un[0])
    print((c10 * calc_dens(un[0]))/1.225)
    shear = sqrt((c10 * calc_dens(un[0]))/1.225) *  Uw(n) # shear velocity
    
    # coefficient times wind velocity squared
    KE = shear *  tau * dt # kinetic energy as function of wind
    
    if ice:
      KE = KE * KEice
    
    date_time = daily_meteo.date
    day_of_year_list = daily_meteo.day_of_year_list
    time_of_day_list = daily_meteo.time_of_day_list
    
    input_data_raw = {'depth':[i for i in range(1,26)],
                             'ShearVelocity_mS-1':np.ones(25)* shear,
                             'ShearStress_Nm-2':np.ones(25)* tau,
                             'day_of_year':np.ones(25) * day_of_year_list[int(n/dt)],
                             'time_of_day':np.ones(25) * time_of_day_list[int(n/dt)],
                             'ice':np.ones(25) * Hi,
                             'snow':np.ones(25) * Hs,
                             'snowice':np.ones(25) * Hsi,
                             'temp_diff02':np.ones(25) * u}
                             #'diffusivity':np.ones(25) * kzn}
    input_mcl = pd.DataFrame(input_data_raw)
    input_columns = ['depth', 'ShearVelocity_mS-1', 'ShearStress_nM-2', 'day_of_year', 'time_of_day', 
                'ice', 'snow', 'snowice', 'temp_diff02']
    
    #scaler = StandardScaler()
    #scaler.fit(input_mcl)
    
    input_data = scaler.transform(input_mcl)    

    input_data_tensor = torch.tensor(input_data, device = torch.device('cpu'))
    
    #breakpoint()
    
    output_tensor = mixing_model(input_data_tensor.float())
    
    output_array = output_tensor.detach().cpu().numpy()
    
 
    #u = output_array * input_std[10] + input_mean[10]
    u = output_array * std_scale + mean_scale
    # breakpoint()
    
    u = u[:,0]

    maxdep = 0
    for dep in range(0, nx-1):
      if dep == 0:
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
             # abs(calc_dens(u[dep+1])- calc_dens(u[dep])))
             abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0])))))
      else:
        PEprior = deepcopy(PE)
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
            # abs(calc_dens(u[dep+1])- calc_dens(u[dep]))) + PEprior
            abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0:(dep+1)])))) + PEprior)
            
      if PE > KE:
        maxdep = dep - 1
        break
      
      maxdep = dep
      
    mix[0,idn] = KE/PE #append(mix, KE/PE)
    therm_z[0,idn] = depth[maxdep] #append(therm.z, maxdep)
    
    if pgdl_mode == 'on':
      um_mix[:, idn] = u

    end_time = datetime.datetime.now()
    print("mixing: " + str(end_time - start_time))

    ## (4) DENSITY INSTABILITIES
    # convective overturn: Convective mixing is induced by an unstable density 
    # profile. All groups of water layers where the vertical density profile is 
    # unstable are mixed with the first stable layer below the unstable layer(s) 
    # (i.e., a layer volume weighed means of temperature and other variables are 
    # calculated for the mixed water column). This procedure is continued until 
    # the vertical density profile in the whole water column becomes neutral or stable.
    start_time = datetime.datetime.now()  
    dens_u = calc_dens(u) 
    diff_dens_u = np.diff(dens_u) 
    diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
    while np.any(diff_dens_u < 0):
      dens_u = calc_dens(u)
      for dep in range(0, nx-1):
        if dens_u[dep+1] < dens_u[dep] and abs(dens_u[dep+1] - dens_u[dep]) >= denThresh:
          u[(dep):(dep+2)] = np.sum(u[(dep):(dep+2)] * volume[(dep):(dep+2)])/np.sum(volume[(dep):(dep+2)])
          break
        
      dens_u = calc_dens(u)
      diff_dens_u = np.diff(dens_u)
      diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
      
    dens_u_n2 = calc_dens(u)
    n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
    if np.max(n2) > 1e-4:
      max_n2 = depth[np.argmax(n2)]
    else:
      max_n2 = np.max(depth)
    mix_z[0, idn] = max_n2
    if pgdl_mode == 'on':
      um_conv[:, idn] = u
    
    end_time = datetime.datetime.now()
    print("convection: " + str(end_time - start_time))
      
    ## (5) ICE FORMATION
    # according to Saloranta & Andersen (2007) and ice growth due to Stefan's law
    # (Leppäranta 1991)
    start_time = datetime.datetime.now()  
    icep  = max(dt_iceon_avg,  (dt/86400))
    x = (dt/86400) / icep
    iceT = iceT * (1 - x) + u[0] * x
    
    K_snow = 2.22362 * (rho_snow/1000)**1.885
    Tice = 0
    
    
    if (iceT <= 0) and Hi < Ice_min and Tair(n) <= 0 and ice == False:
      supercooled = u < 0
      initEnergy = np.sum((0-u[supercooled])*area[supercooled] * dx * Cw)
      
      Hi = Ice_min+(initEnergy/(910*L_ice))/np.max(area)
      
      ice = True
      
      if Hi >= 0:
          
        u[supercooled] = 0
        u[0] = 0
        
    elif ice == True and Hi >= Ice_min:
        Q_surf = (u[0] - 0) * Cw * dx
        u[0] = 0
        
        if Tair(n) > 0:
            Tice = 0
            dHsnew = 0
            
            if (Hs > 0):
                dHs = (-1) * np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice)])
                if (Hs + dHs) < 0:
                    Hi_new = Hi + (Hs + dHs) * (rho_fw/rho_ice)
                else:
                    Hi_new = Hi
            else:
                dHs = 0
                
                Hi_new = Hi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                Hsi = Hsi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                if Hsi <= 0:
                    Hsi = 0
        else:
            if Hs > 0:
                K_snow = 2.22362 * (rho_snow/1000)**(1.885)
                p = (K_ice/K_snow) * (((rho_fw/rho_snow) * Hs ) / Hi)
                dHsi = np.max([0, Hi * (rho_ice/rho_fw -1) + Hs])
                Hsi = Hsi + dHsi

            else:
                p = 1/(10 * Hi)
                dHsi = 0
            
            Tice = (p * 0 + Tair(n)) / (1 + p)
            Hi_new = np.sqrt((Hi + dHsi)**2 + 2 * K_ice/(rho_ice * L_ice)* (0 - Tice) * dt)
            
            # PRECIPITATION
            dHsnew = PP(n) * 1/(1000 * 86400) * dt

            dHs = dHsnew - dHsi * (rho_ice/rho_fw)
            dHsi = 0   

                
        Hi = Hi_new - np.max([0,(Q_surf/(rho_ice * L_ice))])
        
        # if Hi > 2:
        #     print(Hi)
        #     print(meltP * dt * (((1 - IceSnowAttCoeff) * Jsw(n) + (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) + #longwave(emissivity = emissivity, Jlw = Jlw(n)) +
        #                                                                            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
        #                                                                            latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd) + 
        #                                                                            sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)) ))/ (rho_fw * L_ice))
        #     breakpoint()
    
        Q_surf = 0

        Hs = Hs + dHs
        
        

    
    
        if Hi < Hsi:
            Hsi = np.max([0, Hi])
        
        if Hs <= 0:
            Hs = 0
            rho_snow = rho_new_snow
        else:
            rho_snow = rho_snow * (Hs - dHsnew) / Hs + rho_new_snow * dHsnew/Hs
    elif ice and Hi <= Ice_min:
        ice = False
    
    if (ice == False):
        Hi = 0
        Hs = 0
        Hsi = 0
    
    
    Him[0,idn] = Hi
    Hsm[0,idn] = Hs
    Hsim[0,idn] = Hsi
    Ticem[0,idn] = Tice
    

    last_ice = Hi
    last_snow = Hs
    last_snowice = Hsi

    
    n2m[:,idn] = n2
    um[:,idn] = u
    
    if pgdl_mode == 'on':
      um_ice[:, idn] = u
      meteo_pgdl[0, idn] = Tair(n)
      meteo_pgdl[1, idn] = (longwave(cc = CC(n), sigma = sigma, Tair = Tair(n), ea = ea(n), emissivity = emissivity, Jlw = Jlw(n)) -
        backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps))
      meteo_pgdl[2, idn] = latent(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[3, idn] = sensible(Tair = Tair(n), Twater = un[0], Uw = Uw(n), p2 = p2, pa = Pa(n), ea=ea(n), RH = RH(n), A = area, Cd = Cd)
      meteo_pgdl[4, idn] = Jsw(n)
      meteo_pgdl[5, idn] = kd_light
      meteo_pgdl[6, idn] = shear
      meteo_pgdl[7, idn] = tau
      meteo_pgdl[8, idn] = np.nanmax(area)
    
    end_time = datetime.datetime.now()
    print("ice: " + str(end_time - start_time))


  
  bf_sim = np.apply_along_axis(center_buoyancy, axis=1, arr = um.T, depths=depth)
  

  df_z_df_sim = pd.DataFrame({'time': times, 'thermoclineDep': bf_sim})

  df_z_df_sim['epi'] = np.nan
  df_z_df_sim['hypo'] = np.nan
  df_z_df_sim['tot'] = np.nan
  df_z_df_sim['stratFlag'] = np.nan
  for j in range(df_z_df_sim.shape[0]):
    if np.isnan(df_z_df_sim.loc[j, 'thermoclineDep']):
      cur_z = 1
      cur_ind = 0
    else:
      cur_z = df_z_df_sim.loc[j, 'thermoclineDep']
      cur_ind = np.max(np.where(depth < cur_z))
      
    df_z_df_sim.loc[j, 'epi'] = np.sum(um[0:(cur_ind + 1), j] * area[0:(cur_ind+1)]) / np.sum(area[0:(cur_ind+1)])
    df_z_df_sim.loc[j, 'hypo'] = np.sum(um[ cur_ind:, j] * area[cur_ind:]) / np.sum(area[cur_ind:])
    df_z_df_sim.loc[j, 'tot'] = np.sum(um[:,j] * area) / np.sum(area)
    if calc_dens(um[-1,j]) - calc_dens(um[0,j]) >= 0.1 and np.mean(um[:,j]) >= 4:
      df_z_df_sim.loc[j, 'stratFlag'] = 1
    else:
      df_z_df_sim.loc[j, 'stratFlag'] = 0
  #breakpoint()
  dat = {'temp' : um,
          'diff' : kzm,
          'mixing' : mix,
          'buoyancy' : n2m,
          'icethickness' : Hi,
          'iceflag' : ice,
          'icemovAvg' : iceT,
          'supercooled' : supercooled,
          'mixingdepth' : mix_z,
          'thermoclinedepth' : therm_z,
          'endtime' : endTime, 
          'average' : df_z_df_sim,
          'last_ice' : last_ice,
          'last_snow' : last_snow,
          'last_snowice' : last_snowice,
          'density_snow' : rho_snow}
  if pgdl_mode == 'on':
    dat = {'temp' : um,
               'diff' : kzm,
               'mixing' : mix,
               'buoyancy' : n2m,
               'icethickness' : Him,
               'snowthickness' : Hsm,
               'snowicethickness' : Hsim,
               'iceflag' : ice,
               'icemovAvg' : iceT,
               'supercooled' : supercooled,
               'mixingdepth' : mix_z,
               'thermoclinedepth' : therm_z,
               'endtime' : endTime, 
               'average' : df_z_df_sim,
               'temp_initial' : um_initial,
               'temp_heat' : um_heat,
               'temp_diff' : um_diff,
               'temp_mix' : um_mix,
               'temp_conv' : um_conv,
               'temp_ice' : um_ice,
               'meteo_input' : meteo_pgdl,
               'buoyancy_pgdl' : n2_pgdl,
               'heatflux_lwsl' : Qm,
               'heatflux_sw' : Hm,
               'last_ice' : last_ice,
               'last_snow' : last_snow,
               'last_snowice' : last_snowice,
               'density_snow' : rho_snow}
  
  return(dat)

