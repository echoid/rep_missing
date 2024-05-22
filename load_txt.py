import os
import numpy as np
import pandas as pd
import json


def load_data(dataname,mech,rate):
    data = np.load(f"data/{dataname}/feature.npy")
    label = np.load(f"data/{dataname}/label.npy",allow_pickle=False)
    mask = np.load(f"data/{dataname}/{mech}/{rate}.npy")
    data_with_nan = np.where(1-mask, np.nan, data)
     
    with open(f'data/{dataname}/split_index_cv_seed-1_nfold-5.json', 'r') as file:
        index = json.load(file)
    
    return data_with_nan,label,index