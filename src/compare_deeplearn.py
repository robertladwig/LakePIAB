#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:36:04 2023

@author: robert
"""
os.chdir("/home/robert/Projects/LakePIAB/src")
import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt, log, atan, log
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
from oneD_HeatMixing_Functions import get_hypsography, provide_meteorology, initial_profile, run_thermalmodel, run_hybridmodel

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
  
m0_layers = [14, 32, 32, 1]

heating_model = MLP(m0_layers, activation="gelu")
m0_checkpoint = torch.load(m0_PATH, map_location=torch.device('cpu'))
heating_model.load_state_dict(m0_checkpoint)
heating_model = heating_model.to(device)
 
heating_model.train()


data_df = pd.read_csv("./../MCL/02_training/all_data_lake_modeling_in_time.csv")
data_df = data_df.fillna('')
time = data_df['time']
data_df = data_df.drop(columns=['time'])

m0_input_columns = ['depth', 'AirTemp_degC', 'Longwave_Wm-2', 'Latent_Wm-2', 'Sensible_Wm-2', 'Shortwave_Wm-2',
                'lightExtinct_m-1','Area_m2', 
                 'day_of_year', 'time_of_day', 'ice', 'snow', 'snowice', 'temp_initial00']
m0_input_column_ix = [data_df.columns.get_loc(column) for column in m0_input_columns]

training_frac = 0.60
depth_steps = 25
number_days = len(data_df)//depth_steps
n_obs = int(number_days*training_frac)*depth_steps

data_df_scaler = data_df[data_df.columns[m0_input_column_ix]]
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

input_data = train_data[0:25,m0_input_column_ix]



input_data_tensor = torch.tensor(input_data, device = torch.device('cpu'))
    
    #breakpoint()
    
output_tensor = heating_model(input_data_tensor.float())
    
output_array = output_tensor.detach().cpu().numpy()
    

    #u = output_array * input_std[10] + input_mean[10]
u = output_array * std_scale + mean_scale
