
import os
import sys
import numpy as np
import cupy as cp
from numba import cuda, jit

if os.getenv("boostrsa_isRunSource"):
    sys.path.append(os.getenv("boostrsa_source_home"))
    from boostrsa_types import ShrinkageMethod
    from cores.gpu.basic_operations import outer_sum_square, outer_sum
    from cores.gpu.matrix import diag, eyes
    from cores.gpu.basic_operations import scaling
else:    
    from boostrsa.boostrsa_types import ShrinkageMethod
    from boostrsa.cores.gpu.basic_operations import outer_sum_square, outer_sum
    from boostrsa.cores.gpu.matrix import diag, eyes
    from boostrsa.cores.gpu.basic_operations import scaling

def _covariance_eye(residuals: np.ndarray, 
                    threads_per_block = 1024, 
                    dtype = np.float32):
    """
    Computes an optimal shrinkage estimate of a sample covariance matrix as described by the following publication:
    **matrix should be demeaned before!
    
    Ledoit and Wolfe (2004): "A well-conditioned estimator for large-dimensional covariance matrices"
    
    :param residuals: residual data after processing raw data, shape: (#run * #center, #point, #channel)
    :param threads_per_block: #thread per GPU block
    :param dtype: data type for storing array
    """
    # Constant
    n_processing_unit = len(residuals)
    n_point = residuals.shape[1]
    n_channel = residuals.shape[2]
    
    n_block = int(np.ceil(n_processing_unit / threads_per_block))
    
    """
    1. calculate outer product per data of each time
    2. accumulate the outer product result

    GPU memory capacity - (#run * #center, #channel, #channel)
    """
    out_sum_device = cuda.to_device(np.zeros((n_processing_unit, n_channel, n_channel), dtype = dtype))
    outer_sum[n_block, threads_per_block](residuals, out_sum_device)
    outer_sum_result = out_sum_device.copy_to_host()
    del out_sum_device
    cuda.synchronize()
    
    """
    1. calculate outer product per data of each time
    2. accumulate the outer product result with square operation

    GPU memory capacity - (#run * #center, #channel, #channel)
    """
    out_sum_square_device = cuda.to_device(np.zeros((n_processing_unit, n_channel, n_channel), dtype = dtype))
    outer_sum_square[n_block, threads_per_block](residuals, out_sum_square_device)
    outer_sum_square_result = out_sum_square_device.copy_to_host()
    del out_sum_square_device
    cuda.synchronize()
    
    # b2
    s = outer_sum_result / n_point
    s2 = outer_sum_square_result / n_point
    b2 = np.sum(s2 - s * s, axis = (1, 2)) / n_point

    # calculate the scalar estimators to find the optimal shrinkage:
    # m, d^2, b^2 as in Ledoit & Wolfe paper
    # m - shape: (n_processing_unit)
    # d2 - shape: (n_processing_unit)
    # b2 - shape: (n_processing_unit)
    repeat_eyes = np.repeat(np.eye(n_channel)[:, :, np.newaxis], n_processing_unit, axis = 2).T
    
    diag_s = np.diagonal(s, axis1 = 1, axis2 = 2)
    m = (np.sum(diag_s, axis = 1) / n_channel)
    d2 = np.sum((s - m[:, None, None] * repeat_eyes) ** 2, axis = (1, 2))
    
    b2 = np.minimum(d2, b2)
    
    # shrink covariance matrix
    s_shrink = (b2 / d2 * m)[:, None, None] * repeat_eyes + ((d2-b2) / d2)[:, None, None] * s
    
    # correction for degrees of freedom
    dof = n_point - 1
    s_shrink = s_shrink * n_point / dof
    
    return s_shrink

def _covariance_diag(residuals: np.ndarray, 
                     threads_per_block: int = 1024, 
                     dtype = np.float32):
    """
    Calculate covariance 
    
    Schäfer, J., & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and Implications for Functional Genomics.
    
    :param residuals: residual data after processing raw data, shape: (#run * #center, #point, #channel)
    :param threads_per_block: #thread per GPU block
    :param dtype: data type for storing array
    """
    # Constant
    n_processing_unit = len(residuals)
    n_point = residuals.shape[1]
    n_channel = residuals.shape[2]
    
    n_block = int(np.ceil(n_processing_unit / threads_per_block))

    """
    1. calculate outer product per data of each time
    2. accumulate the outer product result

    GPU memory capacity: (shape: #run * #center * #channel * #channel)
    """
    out_sum_device = cuda.to_device(np.zeros((n_processing_unit, n_channel, n_channel), dtype = dtype))
    outer_sum[n_block, threads_per_block](residuals, out_sum_device)
    outer_sum_result = out_sum_device.copy_to_host()
    del out_sum_device
    cuda.synchronize()
    
    """
    1. calculate outer product per data of each time
    2. accumulate the outer product result with square operation

    GPU memory capacity: (shape: #run * #center * #channel * #channel)
    """
    out_sum_square_device = cuda.to_device(np.zeros((n_processing_unit, n_channel, n_channel), dtype = dtype))
    outer_sum_square[n_block, threads_per_block](residuals, out_sum_square_device)
    outer_sum_square_result = out_sum_square_device.copy_to_host()
    del out_sum_square_device
    cuda.synchronize()

    # s
    dof = n_point - 1
    s = outer_sum_result / dof

    """
    Calculate variance per each channel & run

    GPU memory capacity: (shape: #run * #center * #channel * dataType)
    """
    stack_var_device = cuda.to_device(np.zeros((n_processing_unit, n_channel)))
    diag[n_block, threads_per_block](s, stack_var_device)
    stack_var = stack_var_device.copy_to_host()
    del stack_var_device
    cuda.synchronize()
    
    # std
    stack_std = np.sqrt(stack_var)

    # sum mean
    stack_s_mean = outer_sum_result / np.expand_dims(stack_std, 1) / np.expand_dims(stack_std, 2) / (n_point - 1)

    # s2 mean
    stack_s2_mean = outer_sum_square_result / np.expand_dims(stack_var, 1) / np.expand_dims(stack_var, 2) / (n_point - 1)

    # var_hat
    stack_var_hat = n_point / dof ** 2 * (stack_s2_mean - stack_s_mean ** 2)

    # mask
    mask = ~np.eye(n_channel, dtype = bool)

    """
    Scaling

    GPU memory capacity: (shape: #run * #center * #channel * #channel * dataType)
    """
    stack_scaling_mats_device = cuda.to_device(np.zeros((n_processing_unit, n_channel, n_channel), dtype = dtype))
    eyes[n_block, threads_per_block](stack_scaling_mats_device)
    cuda.synchronize()

    stack_lamb_device = np.sum(stack_var_hat[:, mask], axis = 1) / np.sum(stack_s_mean[:, mask] ** 2, axis = 1)
    stack_lamb_device = cp.maximum(cp.minimum(cp.array(stack_lamb_device), 1), 0)
    scaling[n_block, threads_per_block](stack_scaling_mats_device, stack_lamb_device)
    stack_s_shrink = s * stack_scaling_mats_device
    del stack_lamb_device
    cuda.synchronize()
    
    return stack_s_shrink   


