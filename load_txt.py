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

    
        # for fold in index.keys():
        #     fold = index[fold]
        #     # use nan to inpute X data
        #     X[mask == 0] = np.nan

        #     train_index = fold['train_index']
        #     test_index = fold['test_index']

        #     X_train = X[train_index]
        #     X_test = X[test_index]

        #     y_train = y[train_index]
        #     y_test = y[test_index]


        #     # np.savetxt(path+'train_data.txt', X_train, delimiter=',')
        #     # np.savetxt(path+'test_data.txt', X_test, delimiter=',')
        #     # np.savetxt(path+'train_labels.txt', y_train, delimiter=',')
        #     # np.savetxt(path+'test_labels.txt', y_test, delimiter=',')

        #     return X_train, X_test, y_train, y_test, prefilled_path





from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame and the label column is 'target'



def load_data(name):
    label_encoder = LabelEncoder()
    if name == 'banknote':
     with open('datasets/banknote/data_banknote_authentication.txt', 'rb') as f:
        df = pd.read_csv(f, low_memory=False, sep=',')
        X = df.values[:,:-1]
        y = df.values[:,-1]
        y = label_encoder.fit_transform(y)
    elif name == 'california':
        data  = fetch_california_housing()
        X = data["data"]
        y = data["target"]

    elif name == 'climate_model_crashes':
        X,y = fetch_climate_model_crashes()
        y = label_encoder.fit_transform(y)
    elif name == 'concrete_compression':
         X,y = fetch_concrete_compression()
    elif name == 'yacht_hydrodynamics':
         X,y = fetch_yacht_hydrodynamics()
    elif name == 'airfoil_self_noise':
         X,y = fetch_airfoil_self_noise()
    elif name == 'connectionist_bench_sonar':
         X,y = fetch_connectionist_bench_sonar()
         y = label_encoder.fit_transform(y) 
    elif name == 'qsar_biodegradation':
         X,y = fetch_qsar_biodegradation()
         y = label_encoder.fit_transform(y) 
    elif name == 'wine_quality_red':
             X,y = fetch_wine_quality_red()
    elif name == 'wine_quality_white':
             X,y = fetch_wine_quality_white() 
    elif name == 'yeast':
             X,y = fetch_yeast()
             y = label_encoder.fit_transform(y)  

    return X,y




def fetch_climate_model_crashes():
    with open('datasets/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = 0)
        # Ignore the two blocking factor
        X = df.values[:, 2:-1]
        y =  df.values[:, -1]

    return X,y

def fetch_concrete_compression():
    with open('datasets/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        X = df.values[:, :-1]
        y =  df.values[:, -1]
    return X,y


def fetch_yacht_hydrodynamics():
    with open('datasets/yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        X = df.values[:, :-1]
        y =  df.values[:, -1]
    return X,y


def fetch_connectionist_bench_sonar():
    with open('datasets/connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)

        X = df.values[:, :-1].astype('float')
        y =  df.values[:, -1]
    return X,y


def fetch_qsar_biodegradation():
    with open('datasets/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        X = df.values[:, :-1].astype('float')
        y =  df.values[:, -1]

    return X,y

def fetch_yeast():
    with open('datasets/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        X = df.values[:, 1:-1].astype('float')
        y =  df.values[:, -1]

    return X,y


def fetch_wine_quality_red():
    with open('datasets/wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        X = df.values[:, :-1].astype('float')
        y =  df.values[:, -1]
    return X,y

# Dpne!
def fetch_wine_quality_white():
    with open('datasets/wine_quality_white/data.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        X = df.values[:, :-1].astype('float')
        y =  df.values[:, -1]
    return X,y
