
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

def reconstruct_sl_precisionMats(sl_precisions: np.ndarray, n_neighbor: int) -> np.ndarray:
    """
    Reconstruct searchlight precision matrix from 1d(combination(n_neighbor, 2)) into 2d(n_neighbor * n_neighbor)

    :param sl_precisions(shape - #center, #source, #element): array of precision matrices
    :param n_neighbor: a number of neighbor of source from precision matrix
    
    return (shape - #center * n_source, n_spaital_dim, n_spatial_dim)
    """
    n_center, n_source, n_element = sl_precisions.shape
    n_batch = n_center * n_source

    # Indices
    r, c = np.triu_indices(n_neighbor, k = 0)
    off = (r != c)
    
    # Reconstruct matrix
    dummy = np.zeros((n_batch, n_neighbor, n_neighbor))
    packed = sl_precisions.reshape(n_batch, n_element)
    dummy[:, r, c] = packed # Allocate upper triangle elements
    dummy[:, c[off], r[off]] = packed[:, off] # Allocate lower triangle elements
    
    return dummy
