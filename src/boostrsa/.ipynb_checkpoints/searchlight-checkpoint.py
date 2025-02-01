
# Common Libraries
import os
import numpy as np
from numba import cuda, jit
import cupy as cp
import itertools
from tqdm import trange
from rsatoolbox.rdm import concat, RDMs

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
    Calculate precision matrix on each center which is calculated based on neighbor information on each center
    
    :param residuals(np.ndarray - shape: (#run, #point, #channel)): residual array after processing GLM
    :param neighbors(np.ndarray - shape: (#center, #neighbor)): index information about neighbors surrounding each center
    :param n_split_data(int): how many datas to process at once
    :param masking_indexes(np.array- shape: (#channel)): 1d location index converted from 3D brain coordinate (x,y,z)
    :param n_thread_per_block(int): #thread per block
    
    return (np.ndarray), shape: (#channel, #run, combination(#neighbor, 2))
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
    Calculate searchlight crossnobis rdm on each center.
    
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

def calc_sl_precisions(centers, 
                       neighbors,
                       residuals,
                       n_split_data,
                       mask_1d_indexes,
                       save_dir_path,
                       shrinkage_method = ShrinkageMethod.shrinkage_diag,
                       n_thread_per_block = 1024):
    """
    Calculate searchlight precision matrix along multiple difference #neighbor
    precision matrices are saved on save_dir_path per #neighbor
    
    :param centers(np.array): centers, shape: (#center)
    :param neighbors(list - shape: (#center, #neighbor)): index information about neighbors surrounding each center
    :param residuals(np.ndarray - shape: (#run, #point, #channel)): residual array after processing GLM
    :param n_split_data(int): how many datas to process at once
    :param mask_1d_indexes(np.array- shape: (#channel)): 1d location index converted from 3D brain coordinate (x,y,z)
    :param save_dir_path(string): directory path for saving precision matrix result
    :param n_thread_per_block(int): #thread per block
    """
    n_neighbors = np.array([neighbor.shape[-1] for neighbor in neighbors])
    uq_neighbors = np.unique(n_neighbors)
    for n_neighbor in uq_neighbors:
        flags = (n_neighbors == n_neighbor)
        target_centers = centers[flags]
        
        target_neighbors = [ne for flag, ne in zip(flags, neighbors) if flag == True]
        target_neighbors = np.array(target_neighbors)
    
        # Calculate precision matrix (searchlight)
        sl_precisions = calc_sl_precision(residuals = residuals,
                                          neighbors = target_neighbors,
                                          n_split_data = n_split_data,
                                          masking_indexes = mask_1d_indexes,
                                          shrinkage_method = ShrinkageMethod.shrinkage_diag,
                                          n_thread_per_block = n_thread_per_block)
        sl_precisions = np.concatenate(sl_precisions)
    
        save_path = os.path.join(save_dir_path, f"precision_neighbor{n_neighbor}")
        np.savez(save_path, 
                 centers = target_centers, 
                 neighbors = target_neighbors,
                 precision = sl_precisions)
        print(f"save: {save_path}")

def calc_sl_rdm_crossnobises(n_split_data,
                             unique_n_neighbors, 
                             precision_dir_path, 
                             measurements, 
                             masking_indexes, 
                             conditions, 
                             sessions,
                             n_thread_per_block = 1024):
    """
    Calculate searchlight crossnobis rdm on each center.
    However, this function calculate RDM differently if n_neighbor is different
    
    1. load precision matrix for corresponding specific #neighbor on precision_dir_path 
    2. Calculate RDM with searchlight way

    Notice) 
    This function load precision matrix on precision_dir_path.
    So you have to check whether the precision matrices are not overlapped.
    
    :param n_split_data(int): how many datas to process at once
    :param unique_n_neighbors(np.array): unique #neighbor 
    :param precision_dir_path(string): path saved for precision matrix
    :param measurements(np.array): measurment values , shape: (#cond, #channel)
    :param masking_indexes: (np.array) , shape: (#channel) , index of masking brain
    :param conditions(np.array - 1d): condition array corresponding to measurements
    :param sessions(np.array - 1d): session corressponding to conds
    :param n_thread_per_block(int): block per thread

    return (rsatoolbox.rdm.RDMs): RDM matrix about each brain coordinate
    """

    centers = []
    rdms = []
    for n_neighbor in unique_n_neighbors:
        # Load precision matrix
        precision_dataSet_path = os.path.join(precision_dir_path, f"precision_neighbor{n_neighbor}.npz")
        precision_dataSet = np.load(precision_dataSet_path)
        target_centers = precision_dataSet["centers"]
        target_neighbors = precision_dataSet["neighbors"]
        sl_precisions = precision_dataSet["precision"]

        # Calculate RDM with searchlight way
        rdm_crossnobis_gpus, rdm_conds = calc_sl_rdm_crossnobis(n_split_data = n_split_data,
                                                                centers = target_centers,
                                                                neighbors = target_neighbors,
                                                                precs = sl_precisions,
                                                                measurements = measurements,
                                                                masking_indexes = masking_indexes,
                                                                conds = conditions,
                                                                sessions = sessions,
                                                                n_thread_per_block = n_thread_per_block)
        rdm_crossnobis_gpus = np.concatenate(rdm_crossnobis_gpus)
    
        # Make sl_rdms
        rdm_crossnobis_gpus = RDMs(rdm_crossnobis_gpus,
                                       pattern_descriptors = {
                                           "index" : list(np.arange(0, len(rdm_conds))),
                                           "events" : list(rdm_conds),
                                       },
                                       rdm_descriptors = {
                                           "voxel_index" : target_centers,
                                           "index" : np.arange(0, len(target_centers))
                                       })
        rdm_crossnobis_gpus.dissimilarity_measure = "crossnobis"
        

        # Acc
        rdms.append(rdm_crossnobis_gpus)
        centers.append(target_centers)

    # Concat
    rdms = concat(rdms)
    centers = np.concatenate(centers)
    
    # Reorder
    rdms = rdms.subsample(by = "voxel_index", value = np.sort(centers))
    
    return rdms
    