"""
Utility functions for training
"""


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics




def label_distribution_smoothing(y, bins=None, ks=5, sigma=2, normalize=True):
    """
    Return a smoothed label distribution derived by convolving a symetric kernel
    to the empirical label distribution. If bins is None, split the data (y) into bins
    such that each bin corresponds to 1.0 pH unit. Otherwise if bins is an integer, split
    bins into as many bins as is specified.
    See the paper,
    Yang, Zha, Chen, et al, 2021. Delving into deep imbalanced regression.
    Code adapted from https://github.com/YyzHarry/imbalanced-regression
    """
        
    # First split the data into bins of equal width
    if bins==None:
        bins = int(np.ceil(np.max(y) - np.min(y)))  # No. of bins using a width of 1.0 pH units
    bin_freqs, bin_borders = np.histogram(y, range=(min(y), max(y)), bins=bins)
    y_binned = np.zeros(len(y))  # Initialize bin indices to 0s
    for i in range(bins):
        low, high = bin_borders[i], bin_borders[i+1] # Low and high boundaries of bin
        locs = np.logical_and((y >= low), (y<= high)) # Location of values in y within bin
        y_binned[locs] = i

    # Compute kernel window
    half_ks = (ks - 1) // 2
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
    kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) 
    kernel_window /= max(gaussian_filter1d(base_kernel, sigma=sigma))

    # Derive Kernel estimation
    bin_kde = convolve1d(np.array(bin_freqs), weights=kernel_window, mode='constant')
    y_kde = np.array([bin_kde[int(item)] for item in y_binned])

    # Normalize KDE
    if normalize:
        y_kde = y_kde / (np.min(y_kde))
    
    return y_kde




def get_sample_weights(ydata, method='bin_inv', bin_borders=[5,9]):
    """
    Return an array of sample weights computed with different methods.
    Methods:
        'None': No reweighting, weights of ones for all samples
        'bin_inv': Inverse of frequency of three main bins [y<=5, 5<y<9, y>=9]
        'bin_inv_sqrt': Square root of bin_inverse
        'LDS_inv': Inverse of Gaussian kernel density estimate 
                   (label distribution smoothing) of target data derived with 
                   kernel size of 5, standard deviation of 2, and 100 equidistant bins.
        'LDS_inv_sqrt': Square root of LDS_inv
        'LDS_extreme': LDS weights are rescaled so that rare values (y<=5, y>=9) are twice
                       more likely to be selected than normal values (5<y<9).
    """
    
    assert method in ['None', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt',
                      'LDS_extreme']
    
    if method == 'None':
        # Non-uniform weights (no weighting, use data as is)
        weights = np.ones(len(ydata))

    elif method in ['bin_inv', 'bin_inv_sqrt']:
        # Inverse frequency weights (1/n)
        y_binned = np.digitize(ydata, bin_borders)  # Continuous targets binned into categorical
        bin_class, bin_freqs = np.unique(y_binned, return_counts=True)
        inv_freq_dict = dict(zip(bin_class, 1 / bin_freqs)) # Frequency of each bin as dictionary
        weights = np.array([inv_freq_dict[value] for value in y_binned]) # Inverse of frequency as sample weights

    elif method in ['LDS_inv', 'LDS_inv_sqrt', 'LDS_extreme']:
        # Label distribution smoothing (LDS) weights, with Gaussian KDE
        effdist = label_distribution_smoothing(ydata, bins=100) 
        weights = 1 / effdist
        if method == 'LDS_extreme':
            # Scale weights so rare values have double weights
            relevance = np.logical_or(ydata<=5, ydata>=9).astype(int) # Select rare values
            relevance = relevance * (1 - 0.5) + 0.5  # Values to double rare weights
            weights = weights * relevance
            
    if method in ['bin_inv_sqrt', 'LDS_inv_sqrt']:
        weights = np.sqrt(weights)
    
    # Normalize so weights have a mean of 1
    weights = weights / np.mean(weights)
    
    return weights




def sample_hyperparameters(param_space, n=200):
    """
    Return hyperparmeters randomly sampled from a dictionary, where keys are hyperparameter names
    and values are arrays of possible hyperparameters.
    """
    
    params_list = []
    maxsize = int(np.product([len(item) for item in param_space.values()]))
    
    while len(params_list) < min([n, maxsize]):
        sampled_params = {}
        for key,values in param_space.items():
            sampled_params[key] = np.random.choice(values)
        if sampled_params not in params_list:
            params_list.append(sampled_params)
    
    return params_list




def performance(ytrue, ypred, weights, bins=[5,9]):
    '''Return a dictionary of performance metrics evaluated on predictions'''
    
    perf = {}
    ytrue, ypred, weights = np.asarray(ytrue), np.asarray(ypred), np.asarray(weights)

    # Correlation (use resampled data to capture sparse acidic/alkaline regions)
    p = weights / np.sum(weights)
    usize = len(weights)
    rho, r = 0, 0
    iters = 100
    for _ in range(iters):
        locs = np.random.choice(range(len(ytrue)), size=usize, p=p, replace=True) 
        rho += float(spearmanr(ytrue[locs], ypred[locs])[0])
        r += float(pearsonr(ytrue[locs], ypred[locs])[0])
    perf['rho'] = rho / iters
    perf['r'] = r / iters

    # Sample-weighted metrics
    perf['rmse'] = float(metrics.mean_squared_error(ytrue, ypred, sample_weight=weights, squared=False))
    perf['r2'] = float(metrics.r2_score(ytrue, ypred, sample_weight=weights))
                 
    # Classification performance of binned data
    ytrue_binned = np.digitize(ytrue, bins)
    ypred_binned = np.digitize(ypred, bins)
    perf['mcc'] = float(metrics.matthews_corrcoef(ytrue_binned, ypred_binned, sample_weight=weights))
    f1score, auc = [], [], 
    for val in set(ytrue_binned):
        ytrue_sel = (ytrue_binned==val).astype(int)
        ypred_sel = (ypred_binned==val).astype(int)
        f1score.append(float(metrics.f1_score(ytrue_sel, ypred_sel, sample_weight=weights)))
        auc.append(float(metrics.roc_auc_score(ytrue_sel, ypred_sel, sample_weight=weights)))
        perf['f1score_per_bin'] = f1score
        perf['f1score'] = np.mean(f1score)
        perf['auc_per_bin'] = auc
        perf['auc'] = np.mean(auc)
        
    return perf

