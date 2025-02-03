
import numpy as np

def convert_1d_to_symmertic(a_1d, size, k = 0, dtype = np.float32):
    """
    Convert 1d array to symmetric matrix
    
    :param a_1d(1d array): 
    :param size: matrix size
    :param k(int): offset 
    
    return (np.array)
    """

    # put it back into a 2D symmetric array

    X = np.zeros((size,size), dtype = dtype)
    X[np.triu_indices(size, k = 0)] = a_1d
    X = X + X.T - np.diag(np.diag(X))

    return X

def mean_fold_variance(variances, fold_info):
    """
    Calculate fold variacne from fold info
    
    :param variances: variances (#run, #cov.shape)
    :param fold_info(2d array): fold information - [[fold1, fold2], ...]
    
    return (np.array) - (#run * (#runC2), cov.shape)
    """
    n_d = len(variances)
    
    result_variances = []
    for i in range(n_d):       
        for fold1_i, fold2_i in fold_info:            
            cov1 = variances[i][fold1_i]
            cov2 = variances[i][fold2_i]
            
            result_variances.append((cov1 + cov2) / 2)
    
    return np.array(result_variances)


