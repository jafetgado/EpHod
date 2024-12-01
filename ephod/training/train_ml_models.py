"""
Traditional machine learning models
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost

from tqdm import tqdm
import os, subprocess, sys
import importlib

sys.path.append('./ephod/training/')
import trainutils
importlib.reload(trainutils);




WEIGHT_TYPES = ['bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt', 'LDS_extreme']




def get_target_data(path=None):
    '''Return pHopt data as a dictionary of arrays'''

    if path is None: # Else specify to correct path
        path = "./data/pHopt_data.csv"
    df = pd.read_csv(path, index_col=0)
    data = {}
    for key in ['Training', 'Validation', 'Testing']:
        dfsel = df[df['Split']==key]
        keydata = {'accession': dfsel['Accession'].values, 
                    'sequence': dfsel['Sequence'].values,
                    'y': dfsel['pHopt'].values}
        keydata['weights'] = {name : trainutils.get_sample_weights(keydata['y'], name) for name in WEIGHT_TYPES}
        data[key] = keydata
    
    return data



def get_features_data(data, path=None):
    '''Fetch features data and add to data dictionary'''
    
    if path is None:
        path = "data/aac.csv"
    X = pd.read_csv(path, index_col=0)
    for key in data.keys():
        print(key)
        data[key]['X'] = X.loc[data[key]['accession'], :].values    
    
    return data
    


    
class Trainer():
    
    def __init__(self, data, modelname='ridge'):

        self.data = data
        self.modelname = modelname
        self.param_space = self.get_param_space()
        self.sampled_params = trainutils.sample_hyperparameters(self.param_space, n=200)

    
    def get_model(self):
        
        if self.modelname == 'ridge':
            model = Ridge()
        elif self.modelname == 'svr':
            model = SVR()
        elif self.modelname == 'rforest':
            model = RandomForestRegressor(n_jobs=-1)
        elif self.modelname == 'knr':
            model = KNeighborsRegressor(n_jobs=-1)
        elif self.modelname == 'xgboost':
            model = xgboost.XGBRegressor(n_jobs=-1)
        else:
            raise ValueError(f'Unrecognized model: {self.modelname}')
        
        return model

    
    def get_param_space(self):
        '''Define space for hyperparameter search'''
        
        if self.modelname == 'ridge':
            param_space = {'alpha': 10.0 ** np.arange(-8,8+1)}
        elif self.modelname == 'svr':
            param_space = {'kernel': ['poly', 'rbf'], 'gamma': ['scale', 'auto'], 'C': 10. ** np.arange(-5, 5)}
        elif self.modelname == 'rforest':
            param_space = {'n_estimators': [10, 20, 50, 100, 200, 500, 1000], 'criterion': ['mse', 'mae'],  
                           'max_features': [0.25, 0.5, 0.75, None], 'max_samples': [0.25, 0.5, 0.75, None],
                           'max_depth': [5,10,None]}
        elif self.modelname == 'knr':
            param_space = {'n_neighbors':[1, 2, 5, 10, 20, 50, 100], 'weights':['distance', 'uniform']}
        elif self.modelname == 'xgboost': 
            param_space = {'learning_rate': [0.01, 0.5, 0.1, 0.2, None], 'reg_alpha': [0, 0.01, 0.1, 1, 10, 100, None],
                           'reg_lambda': [0.01, 0.1, 1, 10, 100, None], 'max_depth': [3, 5, 7, 9, 11, None],
                           'max_delta_step': [0, 1, 5, 10, None], 'min_child_weight': [1, 3, 5, 7, 10], 
                           'n_estimators':[20, 50, 100, 200, 500, None]}
        param_space['weight_type'] = WEIGHT_TYPES # Add reweighting technique as hyperparameter
        
        return param_space


    def normalize_data(self):
    
        Xtrain, Xval, Xtest = [data[key]['X'] for key in ['Training', 'Validation', 'Testing']]
        ytrain, yval, ytest = [data[key]['y'] for key in ['Training', 'Validation', 'Testing']]
        means, stds = np.mean(Xtrain, axis=0), np.std(Xtrain, axis=0)
        Xtrain, Xval, Xtest = [(item - means) / (stds + 1e-8) for item in [Xtrain, Xval, Xtest]] 
        
        return ((Xtrain, Xval, Xtest), (ytrain, yval, ytest))

    
    def train(self):
        '''Train multiple models with all sampled hyperparameters'''
        
        (Xtrain, Xval, Xtest), (ytrain, yval, ytest) = self.normalize_data()
        best_rmse = 1e10 # Initialize to a large number
        
        for params in tqdm(self.sampled_params):
            
            # Sample weights for training (always use bin_inv in validation/testing)
            trainweights = self.data['Training']['weights'][params['weight_type']]
            valweights, testweights = [self.data[key]['weights']['bin_inv'] for key in ('Validation', 'Testing')]
            # Train
            modelparams = {k:v for k,v in params.items() if k!='weight_type'}
            model = self.get_model()
            model = model.set_params(**modelparams)
            model = model.fit(Xtrain, ytrain, sample_weight=trainweights)
            # Validate
            yvalpred = model.predict(Xval)
            valperf = trainutils.performance(yval, yvalpred, valweights)
            # Test
            ytestpred = model.predict(Xtest)
            testperf = trainutils.performance(ytest, ytestpred, testweights)
            # Store results
            rmse = valperf['rmse']
            if rmse < best_rmse:
                best_rmse = rmse
                best_results = params.copy()
                best_results.update({'valperf': valperf, 'testperf': testperf, 'yvalpred': yvalpred,
                                     'ytestpred': ytestpred, 'model': model})
                self.best_results = best_results
        
        return best_results




if __name__ == '__main__': 

    modelnames = ['ridge', 'svr', 'knr', 'rforest', 'xgboost']
    repnames = ['aac', 'ifeaturepca', 'ifeaturerfe', 'onehot', 'esm1v', 'esm1b', 'prott5', 'progen2', 
                'tranception', 'carp']
    data = get_target_data() # First, download from zenodo
    for modelname in modelnames:
        for repname in repnames:
            # Data for training
            data = get_features_data(data, f"../data/{repname}.csv")
            # Train model
            trainer = Trainer(data, modelname)
            results = trainer.train()
            joblib.dump(results/ f"../data/{repname}-{modelname}.pkl")
