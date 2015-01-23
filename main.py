__author__ = 'anirudha'
import numpy as np
from scipy.io import loadmat
import json

data = loadmat('./Data/subject1_mat/train_subject1_raw01.mat')
print data.keys()
X =  data['X']
y = data['Y']
class Params:
    pass

def load_params():
    json_data = open("settings.json")
    data = json.load(json_data)
    params = Params()
    params.data_dir = data["data_dir"]
    params.fs = data["fs"]
    params.num_chars_train = data["num_trials"]
    params.num_chars_train = data["duration"]
    return params
params = load_params()

class EpochExtractorX:

    """
    For extracting X
    """

    def __init__(self):
        self.fs = 512
        self.num_trials = 15
        self.duration = 15

    def transform(self,X):
        X = np.asarray(X)
        flag = []
        n_samples,n_channels = X.shape
        X_new = np.zeros((self.fs*self.duration,n_channels))
        n_seconds = n_samples/self.fs
        for i in range(self.num_trials):
            flag = X[i*self.fs*self.duration:(i+1)*self.fs*self.duration,:]
            X_new = np.dstack((X_new,flag))
        return X_new[:,:,1:]

    def __repr__(self):
        return "EpochExtracterX"

class EpochExtractorY:
    def __init__(self):
        self.fs = 512
        self.num_trials = 15
        self.duration = 15

    def transform(self,Y):
        Y = np.asarray(Y)
        flag = []
        n_samples = Y.shape
        Y_new = np.zeros((self.fs*self.duration,1))
        for i in range(self.num_trials):
            flag = Y[i*self.fs*self.duration:(i+1)*self.fs*self.duration]
            Y_new = np.hstack((Y_new,flag))
        return Y_new[:,1:]

    def __repr__(self):
        return "EpochExtracterY"



X = EpochExtractorX().transform(X)
y = EpochExtractorY().transform(y)
print y.shape
print X.shape