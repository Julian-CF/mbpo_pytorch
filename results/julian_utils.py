
import torch 
import pandas as pd
import os


def save_metrics(metrics, data_path):
    if os.path.exists(data_path):
        if os.path.exists(data_path + '/metrics.csv'):
            header=False
            prev_epochs = pd.read_csv(data_path + '/metrics.csv')['epoch'].iloc[-1]
            metrics['epoch'] += prev_epochs
        else:
            header=True
        metrics = pd.DataFrame.from_dict(metrics)
        metrics.to_csv(data_path + '/metrics.csv', mode='a', header=header, index=False) #? If already created
    else:
        os.makedirs(data_path)
        metrics = pd.DataFrame.from_dict(metrics)
        metrics.to_csv(data_path + '/metrics.csv', mode='a', index=False)