import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances

def IKFeature(data, Sdata=None, psi=64, t=200, Sp=True):
    if Sdata is None:
        Sdata = data.copy()

    sizeS = Sdata.shape[0]
    sizeN = data.shape[0]
    Feature = None

    for _ in range(t):
        subIndex = check_random_state(None).choice(sizeS, psi, replace=False)
        
        tdata = Sdata[subIndex, :]
        distances = pairwise_distances(tdata, data, metric='euclidean')
        #print(distances)
        nn_indices = np.argmin(distances, axis=0)
        OneFeature = np.zeros((sizeN, psi), dtype=int)
        OneFeature[np.arange(sizeN), nn_indices] = 1
        
        if Feature is None:
            Feature = OneFeature  # Initialize Feature with OneFeature in the first iteration
        else:
            Feature = np.hstack((Feature, OneFeature))  # Add OneFeature to Feature vertically


    return Feature  # sparse matrix
 


def IKSimilarity(data, Sdata=None, psi=64, t=200):
    Feature = IKFeature(data, Sdata, psi, t)
    SimMatrix = np.dot(Feature, Feature.T) / t
    return SimMatrix