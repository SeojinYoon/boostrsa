
# Common Libraries
import numpy as np
from numba import cuda, jit
import cupy as cp
import itertools
from tqdm import trange

# Custom Libraries
from boostrsa.boostrsa_types import ShrinkageMethod
from boostrsa.cores.cpu.matrix import convert_1d_to_symmertic, mean_fold_variance
from boostrsa.cores.cpgpu.stats import _covariance_diag, _covariance_eye
from boostrsa.cores.gpu.mask import set_mask
from boostrsa.cores.gpu.matrix import calc_kernel, rdm_from_kernel

# Functions
def calc_sl_precision(residuals, 
                      neighbors, 
                      n_split_data, 
                      masking_indexes, 
                      n_thread_per_block = 1024,
                      shrinkage_method = "shrinkage_diag"):
    """
    Calculate precision
    
    :param residuals(np.ndarray):  , shape: (#run, #point, #channel)
    :param neighbors(np.ndarray): , shape: (#center, #neighbor)
    :param n_split_data(int): how many datas to process at once
    :param masking_indexes(np.array):  , shape: (#channel) / index of masking brain
    :param n_thread_per_block(int): block per thread
    
    return (np.ndarray), shape: (#channel, #run, combination(#neighbor,2))
    """
    
    n_run = residuals.shape[0]
    n_p = residuals.shape[1]
    n_channel = residuals.shape[-1]
    
    n_center = len(neighbors)
    n_block = int(np.ceil(n_split_data / n_thread_per_block))
    n_neighbor = neighbors.shape[-1]
    r, c = np.triu_indices(n_neighbor, k = 0)
    
    mempool = cp.get_default_memory_pool()
    
    chunk_precisions = []
    for i in trange(0, n_center, n_split_data):
        # select neighbors
        target_neighbors = neighbors[i:i + n_split_data, :]
        len_target = len(target_neighbors)
        
        # output_1d
        mask_out = cuda.to_device(np.zeros((len_target, n_channel)))

        # Make mask - neighbor
        set_mask[n_block, n_thread_per_block](target_neighbors, masking_indexes, mask_out)
        
        # sync
        cuda.synchronize()

        # Apply mask
        cpu_mask = mask_out.copy_to_host()
        masked_residuals = []
        for j in range(len(target_neighbors)):
            masked_residuals.append(residuals[:, :, cpu_mask[j] == 1])
        masked_residuals = np.array(masked_residuals)

        del mask_out
        cuda.defer_cleanup()

        # Calculate demean
        target_residuals = masked_residuals.reshape(-1, n_p, n_neighbor)
        mean_residuals = np.mean(target_residuals, axis = 1, keepdims=1)
        target_residuals = (target_residuals - mean_residuals)

        # Calculate covariance
        if shrinkage_method == ShrinkageMethod.shrinkage_diag:
            covariances = _covariance_diag(target_residuals)
        elif shrinkage_method == ShrinkageMethod.shrinkage_eye:
            covariances = _covariance_eye(target_residuals)

        # Calculate precision matrix
        stack_precisions = cp.linalg.inv(cp.asarray(covariances)).get()
        
        # sync
        cuda.synchronize()
        
        # concat
        stack_precisions = stack_precisions.reshape(len_target, n_run, n_neighbor, n_neighbor)
        stack_precisions = stack_precisions[:, :, r, c]
    
        # add chunk
        chunk_precisions.append(stack_precisions)
        
        # Clean data
        cuda.defer_cleanup()
        mempool.free_all_blocks()
        
    return chunk_precisions

def calc_sl_rdm_crossnobis(n_split_data, 
                           centers, 
                           neighbors, 
                           precs,
                           measurements,
                           masking_indexes,
                           conds, 
                           sessions, 
                           n_thread_per_block = 1024):
    """
    Calculate searchlight crossnobis rdm
    
    :param n_split_data(int): how many datas to process at once
    :param centers(np.array): centers, shape: (#center)
    :param neighbors(np.array): neighbors , shape: (#center, #neighbor)
    :param precs(np.array): precisions , shape: (#channel, #run, #precision_mat_element)
    :param measurements(np.array): measurment values , shape: (#cond, #channel)
    :param masking_indexes: (np.array) , shape: (#channel) , index of masking brain
    :param conds: conds(np.array - 1d)
    :param sessions(np.array - 1d): session corressponding to conds
    :param n_thread_per_block(int): , block per thread
    
    """
    # Data configuration
    n_run = len(np.unique(sessions))
    n_cond = len(np.unique(conds))
    n_dissim = int((n_cond * n_cond - n_cond) / 2)
    n_neighbor = neighbors.shape[-1]
    uq_conds = np.unique(conds)
    n_channel = measurements.shape[-1]
    uq_sessions = np.unique(sessions)
    
    assert n_channel == masking_indexes.shape[0], "n_channel should be same"
    
    # Fold
    fold_info = cuda.to_device(list(itertools.combinations(np.arange(len(uq_sessions)), 2)))
    n_fold = len(fold_info)
    total_calculation = n_split_data * n_fold
    
    # GPU Configuration
    n_block = int(np.ceil(n_split_data / n_thread_per_block))
    n_thread_per_block_2d = int(np.ceil(np.sqrt(n_thread_per_block)))
    block_2ds = (total_calculation // n_thread_per_block_2d, total_calculation // n_thread_per_block_2d)
    thread_2ds = (n_thread_per_block_2d, n_thread_per_block_2d)
    
    # Memory pool
    mempool = cp.get_default_memory_pool()
    
    # Calculation
    rdm_outs = []
    for i in trange(0, len(centers), n_split_data):
        # select neighbors
        target_centers = centers[i:i + n_split_data]
        target_neighbors = neighbors[i:i + n_split_data, :]

        n_target_centers  = len(target_centers)

        # output_1d
        mask_out = cuda.to_device(np.zeros((n_target_centers, n_channel)))

        # Make mask - neighbor
        set_mask[n_block, n_thread_per_block](target_neighbors, masking_indexes, mask_out)
        cuda.synchronize()

        # Apply mask
        cpu_mask = mask_out.copy_to_host()
        masked_measurements = []
        for j in range(n_target_centers):
            masked_measurements.append(measurements[:, cpu_mask[j] == 1])
        masked_measurements = np.array(masked_measurements)
        masked_measurements = cp.asarray(masked_measurements)

        del mask_out
        cuda.defer_cleanup()

        # precision
        prec_mat_shape = int((n_neighbor * n_neighbor - n_neighbor) / 2) + n_neighbor
        target_precs = precs[i:i+n_target_centers].reshape(-1, prec_mat_shape)
        target_precs = np.array([convert_1d_to_symmertic(pre, size = n_neighbor) for pre in target_precs])
        variances = cp.linalg.inv(cp.asarray(target_precs))
        variances = variances.reshape(n_target_centers, n_run, n_neighbor, n_neighbor).get()
        fold_preicions = cp.linalg.inv(cp.asarray(mean_fold_variance(variances, fold_info.copy_to_host())))
        fold_preicions = cuda.to_device(fold_preicions.reshape(n_target_centers, len(fold_info), n_neighbor, n_neighbor).get())
        mempool.free_all_blocks()

        # Avg conds per session
        avg_measurements = []
        avg_conds = []
        for session in uq_sessions:
            filtering_session = sessions == session
            sess_cond = conds[filtering_session]
            sess_measurements = cp.compress(filtering_session, masked_measurements, axis = 1)

            mean_measurments = []
            for cond in uq_conds:
                filtering_cond = sess_cond == cond
                cond_measurments = cp.compress(filtering_cond, sess_measurements, axis = 1)
                mean_cond_measurement = cp.mean(cond_measurments, axis = 1)
                mean_measurments.append(cp.expand_dims(mean_cond_measurement, axis = 1))

                avg_conds.append(cond)

            avg_measurements.append(cp.expand_dims(cp.concatenate(mean_measurments, axis = 1), axis = 1))
        avg_measurements = cp.concatenate(avg_measurements, axis = 1).get()

        avg_conds = np.array(avg_conds)

        mempool.free_all_blocks()

        # make kernel
        avg_measurements = cuda.to_device(avg_measurements)

        matmul1_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_cond, n_neighbor)))
        kernel_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_cond, n_cond)))
        calc_kernel[block_2ds, thread_2ds](avg_measurements, fold_preicions, fold_info, matmul1_out, kernel_out)

        cuda.synchronize()
        del matmul1_out
        cuda.defer_cleanup()

        rdm_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_dissim)))
        rdm_from_kernel[block_2ds, thread_2ds](kernel_out, n_neighbor, rdm_out)

        cuda.synchronize()

        mean_rdms = cp.mean(rdm_out.copy_to_host(), axis = 1)
        rdm_outs.append(mean_rdms)

        del kernel_out
        del rdm_out
        cuda.defer_cleanup()
        
    return rdm_outs, uq_conds
