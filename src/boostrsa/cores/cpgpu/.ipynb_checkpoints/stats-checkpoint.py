
import numpy as np
import cupy as cp
from numba import cuda, jit
from boostrsa.boostrsa_types import ShrinkageMethod
from boostrsa.cores.gpu.basic_operations import outer_sum_square, outer_sum
from boostrsa.cores.gpu.matrix import diag, eyes
from boostrsa.cores.gpu.basic_operations import scaling

def _covariance_eye(residuals, threads_per_block = 1024):
    """
    Computes an optimal shrinkage estimate of a sample covariance matrix as described by the following publication:
    **matrix should be demeaned before!
    
    Ledoit and Wolfe (2004): "A well-conditioned estimator for large-dimensional covariance matrices"
    
    :param residuals(np.ndarray): , shape: (#data, #point, #channel)
    """
    print("shrinakge method:", ShrinkageMethod.shrinkage_eye)
    
    # Constant
    data_len = len(residuals)
    n_point = residuals.shape[1]
    n_channel = residuals.shape[2]
    
    n_block = int(np.ceil(data_len / threads_per_block))
    
    # sum
    out_sum_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))

    # sum square
    out_sum_square_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))
    
    # Calc sum, sum square
    outer_sum[n_block, threads_per_block](residuals, out_sum_device)
    outer_sum_square[n_block, threads_per_block](residuals, out_sum_square_device)

    # b2
    s = out_sum_device.copy_to_host() / n_point
    s2 = out_sum_square_device.copy_to_host() / n_point
    b2 = np.sum(s2 - s * s, axis = (1, 2)) / n_point

    # calculate the scalar estimators to find the optimal shrinkage:
    # m, d^2, b^2 as in Ledoit & Wolfe paper
    # m - shape: (data_len)
    # d2 - shape: (data_len)
    # b2 - shape: (data_len)
    repeat_eyes = np.repeat(np.eye(n_channel)[:, :, np.newaxis], data_len, axis = 2).T
    
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

def _covariance_diag(residuals, threads_per_block = 1024):
    """
    Calculate covariance 
    **matrix should be demeaned before!
    
    Schäfer, J., & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale
    Covariance Matrix Estimation and Implications for Functional Genomics.
    
    :param residuals(np.ndarray): , shape: (#data, #point, #channel)
    """
    print("shrinakge method:", ShrinkageMethod.shrinkage_diag)
    
    # Constant
    data_len = len(residuals)
    n_point = residuals.shape[1]
    n_channel = residuals.shape[2]
    
    n_block = int(np.ceil(data_len / threads_per_block))

    # sum
    out_sum_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))

    # sum square
    out_sum_square_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))

    # Calc sum, sum square
    outer_sum[n_block, threads_per_block](residuals, out_sum_device)
    outer_sum_square[n_block, threads_per_block](residuals, out_sum_square_device)

    # s
    dof = n_point - 1
    s = out_sum_device.copy_to_host() / dof

    # var
    stack_var_device = cuda.to_device(np.zeros((data_len, n_channel)))
    diag[n_block, threads_per_block](s, stack_var_device)

    # std
    stack_std = np.sqrt(stack_var_device)

    # sum mean
    stack_s_mean = out_sum_device / np.expand_dims(stack_std, 1) / np.expand_dims(stack_std, 2) / (n_point - 1)

    # s2 mean
    stack_s2_mean = out_sum_square_device / np.expand_dims(stack_var_device, 1) / np.expand_dims(stack_var_device, 2) / (n_point - 1)

    # var_hat
    stack_var_hat = n_point / dof ** 2 * (stack_s2_mean - stack_s_mean ** 2)

    # mask
    mask = ~np.eye(n_channel, dtype=bool)

    # lamb
    stack_lamb_device = np.sum(stack_var_hat[:, mask], axis = 1) / np.sum(stack_s_mean[:, mask] ** 2, axis = 1)
    stack_lamb_device = cp.maximum(cp.minimum(cp.array(stack_lamb_device), 1), 0)

    # Scaling
    stack_scaling_mats_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))
    eyes[n_block, threads_per_block](stack_scaling_mats_device)

    scaling[n_block, threads_per_block](stack_scaling_mats_device, stack_lamb_device)
    stack_s_shrink = s * stack_scaling_mats_device
    
    return stack_s_shrink   


