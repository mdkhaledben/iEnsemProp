#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:46:18 2022

@author: khaled
"""

# fixed randomization seed for reproducible results
import numpy as np
np.random.seed(25)
import torch
torch.manual_seed(25) # https://pytorch.org/docs/stable/notes/randomness.html
import random
random.seed(25)

import os
import pandas  as pd
from datetime import datetime
import time
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force execution in cpu, even if gpu is available

import sys
sys.path.append('/home/khaled/Dropbox/Water/MultistepForecast/codes/forecastnet/pytorch')
# sys.path.append('/lscratch/s5084400/WaterProject/MultistepForecast/codes/forecastnet/pytorch')
# sys.path.append('/lscratch/s5084397/WaterProject/MultistepForecast/codes/forecastnet/pytorch')

from forecastNet import forecastNet
from train import train
from evaluate import evaluate, get_prediction_only
from dataHelpers import *
from calculateError_custom import *

#model_type = 'dense' #'dense2' or 'conv', or 'conv2'
model_type = 'dense2'


#%% Globals
dataset_name = 'BurnettRiver'
target_series_name = 'do'

max_step_ahead = 48
period = max_step_ahead  # period is the natural seasonal period, e.g. 24 hours in temperature change


#%% Data Paths

data_dir = '/home/khaled/Water/MultistepForecast'
# data_dir = '/lscratch/s5084400/Water/MultistepForecast'
# data_dir = '/lscratch/s5084397/Water/MultistepForecast'

models_dir = os.path.join(data_dir, 'models', dataset_name, target_series_name, 'ForecastNet', model_type+'_models')
if not os.path.exists(models_dir): os.makedirs(models_dir)

results_dir = os.path.join(data_dir, 'results', dataset_name, target_series_name, 'ForecastNet', model_type+'_results')
if not os.path.exists(results_dir): os.makedirs(results_dir)

TRAIN_FILENAME = 'trainset.tsv'
TEST_FILENAME = 'testset.tsv'


train_set = pd.read_csv(os.path.join(data_dir, 'results', dataset_name, target_series_name, TRAIN_FILENAME), sep='\t', parse_dates=['datetime'], infer_datetime_format=True, index_col=['datetime'], low_memory=False,)
test_set = pd.read_csv(os.path.join(data_dir, 'results', dataset_name, target_series_name, TEST_FILENAME), sep='\t', parse_dates=['datetime'], infer_datetime_format=True, index_col=['datetime'], low_memory=False,)

dataset = pd.concat([train_set, test_set])[target_series_name].to_numpy().reshape(-1, 1)
test_start_idx = dataset.shape[0] - len(test_set) # starting index of test set in combine dataset
data_scale = {'data_max' : np.max(dataset), 'data_min' : np.min(dataset)} # to be used in data scaling and reverse scaling


train_x, train_y, test_x, test_y, valid_x, valid_y, period = prep_train_test_validation(dataset, period, test_start_idx, n_seqs = 1, do_scaling=True, time_major=True)


# Model parameters
#model_type = 'dense2' #'dense' or 'conv', 'dense2' or 'conv2'
in_seq_length = 2 * period
out_seq_length = period
hidden_dim = 24
input_dim = 1
output_dim = 1
learning_rate = 0.0001
n_epochs= 100
batch_size = 16


# Initialise model
model_file = os.path.join(models_dir, dataset_name +'_' + target_series_name +'_fnet_'+ model_type +'.pt')
fcstnet = forecastNet(in_seq_length=in_seq_length, out_seq_length=out_seq_length, input_dim=input_dim,
                        hidden_dim=hidden_dim, output_dim=output_dim, model_type = model_type, batch_size = batch_size,
                        n_epochs = n_epochs, learning_rate = learning_rate, save_file = model_file)

# train
start_time = time.time()
training_costs, validation_costs = train(fcstnet, train_x, train_y, valid_x, valid_y, restore_session=False)
end_time = time.time()
print('Time (minutes) taken to model training: ', (end_time-start_time)/60)


y_pred = get_prediction_only(fcstnet, test_x, test_y)


#%% Evaluate the model

# forecastnet's default input-outputs shape [out_seq_length, n_samples, n_features=1]
# reshape ground truth and forecastnet's predictions in shape [n_samples, out_seq_length] for easier metric calculation
y_test = np.copy(test_y)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
y_test = np.transpose(y_test, (1, 0))
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
y_pred = np.transpose(y_pred, (1, 0))


mase_fnet, smape_fnet, _ = calculate_error_stepaheadwise(y_test, y_pred) #metric value for each step ahead
mae_fnet, rmse_fnet, mape_fnet = calculate_scale_dependent_error(y_test, y_pred, data_scale) # these metrics not dependent on samplewise or stepwise

df_error_scores_stepwise = pd.DataFrame({'step_ahead' : np.arange(max_step_ahead)+1,
                                         'mae' : mae_fnet,
                                         'rmse' : rmse_fnet,
                                         'mase' : mase_fnet.reshape(mase_fnet.shape[0]),
                                         'smape' : smape_fnet.reshape(smape_fnet.shape[0]),
                                         'mape' : mape_fnet,
                                         })

df_error_scores_stepwise.to_csv(os.path.join(results_dir, 'Stepwise_errorscores_forecastnet_' + 'inws_' + str(in_seq_length) + '_outws_' + str(out_seq_length) + '.csv'), sep='\t', index=False, float_format='%.5f') # list of metric scores for each step ahead       

np.savetxt(os.path.join(results_dir, 'forecast_forecastnet'+ '_inws_' + str(in_seq_length) + '_outws_' + str(out_seq_length) +'.csv'), y_pred, delimiter=',', fmt='%.5f')

avg_error_scores = pd.DataFrame()
avg_error_scores = avg_error_scores.append({'lag_window_size' : in_seq_length,
                        'mae' : np.mean(mae_fnet),
                        'rmse' : np.mean(rmse_fnet),
                        'mape' : np.mean(mape_fnet),
                        'mase' : np.mean(mase_fnet),
                        'smape' : np.mean(smape_fnet),
                        }, ignore_index = True)

avg_error_scores.to_csv(os.path.join(results_dir, 'Avg_errorscores_forecastnet.tsv'), sep='\t', index=False, float_format='%.5f')

print('fNET run complete:  ' + model_type + '  ' + dataset_name + ' - ' + target_series_name)
