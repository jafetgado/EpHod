"""Defining hyperparameter space"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR




space = {}

#======================================#
# Traditional machine learning models
#======================================#

# Ridge regression 
space['ridge'] = {
    'alpha': list(np.arange(-8,8+1)), 
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme']
    }

space['rforest'] =  {
    'n_estimators': [10, 20, 50, 100, 200, 500, 1000],
    'criterion': ['mse', 'mae'],
    'max_features': [0.25, 0.5, 0.75, None],
    'max_samples': [0.25, 0.5, 0.75, None],
    'max_depth': [5,10,None], 
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme']
    }


space['svr'] = {
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'],
    'C': 10. ** np.arange(-5, 6),
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme']
    }

space['knr'] = {
    'k': [1, 2, 5, 10, 20, 50, 100],
    'weightby': ['uniform', 'distance'],
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme']
    }
    


#======================================#
# Deep learning models
#======================================#
space['fnn'] = {
    'input_dim':[20480], 
    'hidden_dims':  [
        [128], [256], [512], [1024], 
        [128]*2, [256]*2, [512]*2, [1024]*2, 
        [128]*3, [256]*3, [512]*3, [1024]*3
        ],
    'dropout': [0, 0.1, 0.25, 0.33, 0.5], 
    'activation': ['relu', 'leaky_relu', 'elu', 'gelu'],
    'residual': [True, False],
    'random_seed': [0],
    'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    'l2reg': [0, 1e-5, 1e-4, 1e-3, 1e-2],
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                  'LDS_extreme'],
    }


space['cnn'] = {
    'input_channel': [20],
    'input_length': [1024],
    'start_conv_channel': [32], 
    'num_conv_layers': [2,3,4,5,6,7,8],
    'kernel_size': [3,5,7],
    'conv_dropout': [0, 0.1, 0.25, 0.33, 0.5],
    'pooltype': ['max', 'average'],
    'dense_dim': [32,64,128,256,512,1024],
    'num_dense_layers': [0, 1, 2, 3, 4, 6, 8],
    'dense_dropout': [0, 0.1, 0.25, 0.33, 0.5],
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
    'random_seed': [0],
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme'],
    'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    'l2reg': [0, 1e-5, 1e-4, 1e-3, 1e-2]
    }


space['dcnn'] = {
    
    'input_channel': [20],
    'conv_channel': [24, 32, 48, 64, 96, 128],
    'num_blocks': [1,2,3,4,6],
    'layers_per_block': [10],
    'block_dropout': [0, 0.1, 0.25, 0.33, 0.5],
    'attention_dim': [32, 64, 128, 256, 512, 1024],
    'attention_kernel_size': [3, 5, 7, 9, 11, 13],
    'attention_dropout': [0, 0.1, 0.25, 0.33, 0.5],
    'residual': [True],
    'perceptive': [True],
    'dense_dim': [32, 64, 128, 256, 512, 1024],
    'num_dense_layers': [0, 1, 2, 3, 4, 6],
    'dense_dropout': [0, 0.1, 0.25, 0.33, 0.5],
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
    'random_seed': [0],
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme'],
    'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    'l2reg': [0, 1e-5, 1e-4, 1e-3, 1e-2]
    }


space['rnn'] = {
    'input_channel': [20], 
    'input_length': [1024],
    'gru_dim': [64, 128, 256, 512, 1024],
    'conv_downsample': [1, 2, 4, 8], 
    'conv_dropout': [0, 0.1, 0.25, 0.33, 0.5], 
    'dense_dim': [64, 128, 256, 512, 1024], 
    'dense_dropout': [0, 0.1, 0.25, 0.33, 0.5], 
    'activation': ['relu', 'leaky_relu', 'elu', 'gelu'],
    'random_seed': [0],
    'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme'],
    'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    'l2reg': [0, 1e-5, 1e-4, 1e-3, 1e-2]
    }


space['lat'] = {    
        'input_dim': [1280],
        'attention_dim': [1280],
        'kernel_size': [3, 5, 7, 9, 11, 13, 15],
        'conv_dropout': [0, 0.1, 0.25, 0.33, 0.5], 
        'perceptive': [False],
        'dense_dim': [32, 64, 128, 256, 512, 1024],
        'dense_dropout': [0, 0.1, 0.25, 0.33, 0.5],
        'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
        'random_seed': [0],
        'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                          'LDS_extreme'],
        'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 5e-3],
        'l2reg': [0, 1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        }


space['pat'] = {
        'input_dim': [1280],
        'attention_dim': [1280],
        'kernel_size': [3, 5, 7, 9, 11, 13, 15],
        'conv_dropout': [0, 0.1, 0.25, 0.33, 0.5], 
        'perceptive': [True],
        'dense_dim': [32, 64, 128, 256, 512, 1024],
        'dense_dropout': [0, 0.1, 0.25, 0.33, 0.5],
        'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
        'random_seed': [0],
        'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                          'LDS_extreme'],
        'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 5e-3],
        'l2reg': [0, 1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        }


space['dcat'] =  {
        'input_dim': [1280],
        'input_len': [1024], 
        'attention_kernel_size': [3, 5, 7, 9, 11, 13],
        'conv_dim': [128, 256, 512], 
        'conv_kernel_size': [3], 
        'conv_dropout': [0, 0.1, 0.25, 0.33, 0.5], 
        'dilated_layers': [9], 
        'dense_dim': [256, 512, 1024], 
        'dense_layers': [1], 
        'dense_dropout': [0, 0.1, 0.25, 0.33, 0.5],
        'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
        'random_seed': [0],
        'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                          'LDS_extreme'],
        'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 5e-3],
        'l2reg': [0, 1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        }

        
space['rlat'] = {
        'dim': [1280], 
        'kernel_size': [3, 5, 7, 9, 11, 13, 15],
        'dropout': [0, 0.1, 0.25, 0.33, 0.5], 
        'res_blocks': [1, 2, 4], 
        'activation': ['relu', 'leaky_relu', 'elu', 'gelu'],
        'random_seed': [0],
        'sample_method': ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                          'LDS_extreme'],
        'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 5e-3],
        'l2reg': [0, 1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        }


def getgridsize(space):
    size = 1
    for k, v in space.items():
        size *= len(v)
    return size


for key, value in space.items():
    size = getgridsize(value)
    if size < 1e4:
        print(f"{key}: {size}")
    else:
        print(f"{key}: {size:.1e}")
        
