import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KDTree


def IKFeature(data, Sdata=None, psi=64, t=200, Sp=True):
    if Sdata is None:
        Sdata = data.copy()

    sizeS = Sdata.shape[0]
    sizeN = data.shape[0]
    Feature = None

    for _ in range(t):
        subIndex = check_random_state(None).choice(sizeS, psi, replace=False)
        
        tdata = Sdata[subIndex, :]
#         distances = pairwise_distances(tdata, data, metric='euclidean')
#         #print(distances)
#         nn_indices = np.argmin(distances, axis=0)
        
        tree = KDTree(tdata) 
        dist, nn_indices = tree.query(data, k=1)
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




class Isolation_Kernal:
    def __init__(self, psi=64, t=200):
        self.t = t  # attribute: t
        self.psi = psi    # attribute: t
        self.tree_list = []

    def build_tree(self,Sdata):
        t = self.t
        psi = self.psi
        
        sizeS = Sdata.shape[0]

        tree_list = []
        for _ in range(t):
            subIndex = check_random_state(None).choice(sizeS, psi, replace=False)

            tdata = Sdata[subIndex, :]
            #distances = pairwise_distances(tdata, data, metric='euclidean')
            #nn_indices = np.argmin(distances, axis=0)
            tree = KDTree(tdata) 

            tree_list.append(tree)
            
        self.tree_list = tree_list
        
        
    def build_feature(self,data):
        t = self.t
        psi = self.psi
        tree_list = self.tree_list
        
        sizeN = data.shape[0]
        Feature  = None

        for _ in range(t):
            tree = tree_list[_]
            dist, nn_indices = tree.query(data, k=1)        

            OneFeature = np.zeros((sizeN, psi), dtype=int)
            OneFeature[np.arange(sizeN), nn_indices] = 1

            if Feature is None:
                Feature = OneFeature  # Initialize Feature with OneFeature in the first iteration
            else:
                Feature = np.hstack((Feature, OneFeature))  # Add OneFeature to Feature vertically

        return Feature  # sparse matrix
    
    def cal_similarity(self,feature1,feature2):
        return np.dot(feature1, feature2.T) / self.t