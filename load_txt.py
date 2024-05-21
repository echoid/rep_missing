import os
import numpy as np
import pandas as pd
import json

def load_txt(data,types,para,full_norm):

    X, y = load_data(data)
    if full_norm == "norm":
        X = np.load('datasets/{}/{}_norm.npy'.format(data,data))


    # for original data
    path = 'datasets/{}/{}/{}'.format(data,types,para)
    mask = np.load('{}.npy'.format(path))
    X[mask == 0] = np.nan

    # for prefilled data
    prefilled_path = 'prefilled_data/{}/{}/{}'.format(data,types,para)

    with open('datasets/{}/split_index_cv_seed-1_nfold-5.json'.format(data), 'r') as file:
        index = json.load(file)

    return X, y, index, prefilled_path


def load_data(dataname,mech,rate):
    data = np.load(f"data/{dataname}/feature.npy")
    label = np.load(f"data/{dataname}/label.npy",allow_pickle=False)
    mask = np.load(f"data/{dataname}/{mech}/{rate}.npy")
    data_with_nan = np.where(1-mask, np.nan, data)
     
    with open(f'data/{dataname}/split_index_cv_seed-1_nfold-5.json', 'r') as file:
        index = json.load(file)
    
    return data,label,index