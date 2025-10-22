
# Common Libraries
import re
import os
import sys
import math
import itertools
import cupy as cp
import numpy as np
import pandas as pd
from glob import glob
from tqdm import trange
from pathlib import Path
from numba import cuda, jit
from itertools import combinations, product
from rsatoolbox.rdm import concat, RDMs

# Custom Libraries
if os.getenv("boostrsa_isRunSource"):
    sys.path.append(os.getenv("boostrsa_source_home"))
    from boostrsa_types import ShrinkageMethod
    from cores.cpu.matrix import convert_1d_to_symmertic, mean_fold_variance, reconstruct_sl_precisionMats
    from cores.cpu.mask import set_mask_cpu
    from cores.gpu.mask import set_mask_gpu
    from cores.cpgpu.stats import _covariance_diag, _covariance_eye
    from cores.gpu.matrix import calc_kernel, rdm_from_kernel, denoise, calc_sqrtMat
    from cores.gpu.matrix import differentiate_measurements, calc_cv_distance
else:
    from boostrsa.boostrsa_types import ShrinkageMethod
    from boostrsa.cores.cpu.matrix import convert_1d_to_symmertic, mean_fold_variance, reconstruct_sl_precisionMats
    from boostrsa.cores.cpu.mask import set_mask_cpu
    from boostrsa.cores.gpu.mask import set_mask_gpu
    from boostrsa.cores.cpgpu.stats import _covariance_diag, _covariance_eye
    from boostrsa.cores.gpu.matrix import calc_kernel, rdm_from_kernel, denoise, calc_sqrtMat
    from boostrsa.cores.gpu.matrix import differentiate_measurements, calc_cv_distance
    
# Functions
def calc_sl_precision(residuals: np.ndarray, 
                      neighbors: np.ndarray, 
                      n_split_data: int, 
                      masking_indexes: np.array, 
                      n_thread_per_block: int = 1024,
                      shrinkage_method: str = "shrinkage_diag",
                      dtype = np.float32):
    """
    Calculate precision matrix on each center which is calculated based on neighbor information on each center
    
    :param residuals(shape: (#run, #point, #channel)): residual array after processing GLM
    :param neighbors(shape: (#center, #neighbor)): index information about neighbors surrounding each center
    :param n_split_data: how many datas to process at once
    :param masking_indexes(shape: (#channel)): 1d location index converted from 3D brain coordinate (x,y,z)
    :param n_thread_per_block: #thread per block
    :param dtype: data type for storing array
    
    return (np.ndarray), shape: (#center, #run, combination(#neighbor, 2))
    """
    if residuals.dtype != dtype:
        residuals = residuals.astype(dtype)
    
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
        """
        Masks are made by selected centers' neighbor

        GPU memory capacitiy: (#selected_center, #channel)
        """
        target_neighbors = neighbors[i:i + n_split_data, :]
        n_target_center = len(target_neighbors)
        
        # Apply mask
        mask = set_mask_cpu(target_neighbors, masking_indexes)
        masked_residuals = np.empty((n_target_center, n_run, n_p, n_neighbor), dtype = residuals.dtype)
        for j, m in enumerate(mask):
            masked_residuals[j] = residuals[:, :, m == 1]
    
        # Calculate demean
        target_residuals = masked_residuals.reshape(-1, n_p, n_neighbor)
        mean_residuals = np.mean(target_residuals, axis = 1, keepdims=1)
        target_residuals = (target_residuals - mean_residuals)

        # Calculate covariance
        if shrinkage_method == ShrinkageMethod.shrinkage_diag:
            covariances = _covariance_diag(target_residuals, dtype = dtype)
        elif shrinkage_method == ShrinkageMethod.shrinkage_eye:
            covariances = _covariance_eye(target_residuals, dtype = dtype)

        # Calculate precision matrix
        stack_precisions = cp.linalg.inv(cp.asarray(covariances)).get()
        del covariances
        
        # sync
        cuda.synchronize()
        
        # concat
        stack_precisions = stack_precisions.reshape(n_target_center, n_run, n_neighbor, n_neighbor)
        stack_precisions = stack_precisions[:, :, r, c]
    
        # add chunk
        chunk_precisions.append(stack_precisions)
        
        # Clean data
        cuda.defer_cleanup()
        mempool.free_all_blocks()

    return np.concatenate(chunk_precisions, axis = 0).astype(dtype)

def calc_sl_rdm_crossnobis(n_split_data: int, 
                           centers: np.array, 
                           neighbors: np.array, 
                           precs: np.array,
                           measurements: np.array,
                           masking_indexes: np.array,
                           conds: np.array, 
                           sessions: np.array, 
                           n_thread_per_block: int = 1024,
                           dtype = np.float32):
    """
    Calculate searchlight crossnobis rdm on each center.
    
    :param n_split_data: how many datas to process at once
    :param centers(shape: (#center)): centers, 
    :param neighbors(#center, #neighbor): neighbors , shape: 
    :param precs(shape: (#channel, #run, #precision_mat_element)): precisions 
    :param measurements(shape: (#data, #channel)): measurment values
    :param masking_indexes(shape: #channel): index of masking brain
    :param conds(shape: #data): condition per data
    :param sessions(shape: #data): session corressponding to conds
    :param n_thread_per_block: #thread per GPU block
    :param dtype: data type for storing array
    """
    if measurements.dtype != dtype:
        measurements = measurements.astype(dtype)
        precs = precs.astype(dtype)
        
    # Data configuration
    n_run = len(np.unique(sessions))
    n_cond = len(conds)
    n_unique_cond = len(np.unique(conds))
    n_dissim = int((n_unique_cond * n_unique_cond - n_unique_cond) / 2)
    n_neighbor = neighbors.shape[-1]
    uq_conds = np.unique(conds)
    n_channel = measurements.shape[-1]
    uq_sessions = np.unique(sessions)
    
    assert n_channel == masking_indexes.shape[0], "n_channel should be same"
    
    # Fold
    fold_info = cuda.to_device(list(combinations(np.arange(len(uq_sessions)), 2)))
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

        # Apply mask
        mask = set_mask_cpu(target_neighbors, masking_indexes)
        masked_measurements = np.empty((n_split_data, n_cond, n_neighbor), dtype = dtype)
        for j, m in enumerate(mask):
            masked_measurements[j] = measurements[:, m == 1]
        masked_measurements = cp.asarray(masked_measurements)

        """
        1. Convert precision matrix to covariance matrix
        GPU memory capacitiy: (#selected_center, #run, #channel, #channel)

        2. Mean covariance between two runs
        GPU memory capacity: (#center, #run, #channel, #channel)
        """
        prec_mat_shape = int((n_neighbor * n_neighbor - n_neighbor) / 2) + n_neighbor
        target_precs = precs[i:i+n_target_centers].reshape(-1, prec_mat_shape)
        target_precs = np.array([convert_1d_to_symmertic(pre, size = n_neighbor, dtype = dtype) for pre in target_precs])
        variances = cp.linalg.inv(cp.asarray(target_precs))
        variances = variances.reshape(n_target_centers, n_run, n_neighbor, n_neighbor).get()
        mempool.free_all_blocks()

        """
        Calculate mean precision matrix between two runs
        """
        fold_preicions = cp.linalg.inv(cp.asarray(mean_fold_variance(variances, fold_info.copy_to_host()))).get()
        mempool.free_all_blocks()
        
        fold_preicions = cuda.to_device(fold_preicions.reshape(n_target_centers, len(fold_info), n_neighbor, n_neighbor))

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

        matmul1_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_unique_cond, n_neighbor), dtype = dtype))
        kernel_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_unique_cond, n_unique_cond), dtype = dtype))
        calc_kernel[block_2ds, thread_2ds](avg_measurements, fold_preicions, fold_info, matmul1_out, kernel_out)

        cuda.synchronize()
        del matmul1_out
        cuda.defer_cleanup()

        rdm_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_dissim), dtype = dtype))
        rdm_from_kernel[block_2ds, thread_2ds](kernel_out, n_neighbor, rdm_out)

        cuda.synchronize()

        mean_rdms = cp.mean(rdm_out.copy_to_host(), axis = 1)
        rdm_outs.append(mean_rdms)

        del kernel_out
        del rdm_out
        cuda.defer_cleanup()
        
    return rdm_outs, uq_conds
    
def calc_sl_precisions(centers: np.array, 
                       neighbors: list,
                       residuals: np.ndarray,
                       prec_types: np.ndarray,
                       n_split_data: int,
                       mask_1d_indexes: np.array,
                       save_dir_path: str,
                       shrinkage_method: ShrinkageMethod = ShrinkageMethod.shrinkage_diag,
                       n_thread_per_block: int = 1024,
                       dtype = np.float32) -> dict:
    """
    Calculate searchlight precision matrix along multiple difference #neighbor
    precision matrices are saved on save_dir_path per #neighbor
    
    :param centers(shape: (#center)): centers, 
    :param neighbors(shape: (#center, #neighbor)): index information about neighbors surrounding each center
    :param residuals(shape: (#session, #point, #channel)): residual array after processing GLM
    :param prec_types(shape: #session): session types corresponding to each residual element
    :param n_split_data: how many datas to process at once
    :param mask_1d_indexes(shape: (#channel)): 1d location index converted from 3D brain coordinate (x,y,z)
    :param save_dir_path: directory path for saving precision matrix result
    :param shrinkage_method: shrinkage method
    :param n_thread_per_block: #thread per block
    :param dtype: data type for storing array

    return {
        "#neighbor{number}" : {
            "path" : saved path
            "center_indices" : center 1d indicies,
            "neighbor_indices" : neighbor 1d indices,
            "prec_types" : precision types
        }
    }
    """
    # Neighbor info
    n_neighbors = np.array([neighbor.shape[-1] for neighbor in neighbors])
    uq_neighbors = np.unique(n_neighbors)

    # Result
    result_info = {}
    
    # Investigate whether save_dir_path has already precision matrix
    paths = [os.path.join(save_dir_path, f"precision_neighbor{n_neighbor}.npz") for n_neighbor in uq_neighbors]
    path_forMapping = []
    for n_neighbor, path in zip(uq_neighbors, paths):
        if os.path.exists(path):
            precision_dataSet = np.load(path)
            target_centers = precision_dataSet["centers"]
            target_neighbors = precision_dataSet["neighbors"]
            prec_types = precision_dataSet["prec_types"]
            
            result_info[f"#neighbor{n_neighbor}"] = {
                "path" : path,
                "center_indices" : target_centers,
                "neighbor_indices" : target_neighbors,
                "prec_types" : prec_types,
            }

            print(f"already exist: {path}")
        else:
            path_forMapping.append((n_neighbor, path))

    # Calculate precision matrix using searchlight method per #neighbor
    for n_neighbor, save_path in path_forMapping:
        flags = (n_neighbors == n_neighbor)
        target_centers = centers[flags]
        
        target_neighbors = [ne for flag, ne in zip(flags, neighbors) if flag == True]
        target_neighbors = np.array(target_neighbors)
    
        # Calculate precision matrix (searchlight)
        sl_precisions = calc_sl_precision(residuals = residuals,
                                          neighbors = target_neighbors,
                                          n_split_data = n_split_data,
                                          masking_indexes = mask_1d_indexes,
                                          shrinkage_method = shrinkage_method,
                                          n_thread_per_block = n_thread_per_block,
                                          dtype = dtype)

        # Save
        np.savez(save_path, 
                 centers = target_centers, 
                 neighbors = target_neighbors,
                 precision = sl_precisions,
                 prec_types = prec_types)
        print(f"save: {save_path}")
        result_info[f"#neighbor{n_neighbor}"] = {
            "path" : save_path,
            "center_indices" : target_centers,
            "neighbor_indices" : target_neighbors,
            "prec_types" : prec_types,
        }
        
    return result_info

def calc_sqrt_precisions(precision_paths: list, 
                         save_dir_path: str,
                         n_batch: int,
                         dtype = np.float32):
    """
    Calculat square root about searchlight precision matrix
    & Save the results on the save_dir_path
    
    :param precision_paths(element: str): precision matrix paths
    :param save_dir_path: directory path for saving sqrt precision
    :param n_batch: how many datas to process at once
    
    return {
        "#neighbor{number}" : {
            "path" : saved path
            "center_indices" : center 1d indicies,
            "neighbor_indices" : neighbor 1d indices,
            "prec_types" : precision types
        }
    }
    """
    # Result
    result_info = {}
    
    # Investigate whether save_dir_path has already precision matrix
    sqrt_paths = [os.path.join(save_dir_path, "sqrt" + "_" + Path(path).name) for path in precision_paths]
    path_forMapping = []
    for precision_path, sqrt_path in zip(precision_paths, sqrt_paths):
        if os.path.exists(sqrt_path):
            precision_dataSet = np.load(sqrt_path)
            target_centers = precision_dataSet["centers"]
            target_neighbors = precision_dataSet["neighbors"]
            prec_types = precision_dataSet["prec_types"]
            n_neighbor = target_neighbors.shape[1]
            
            result_info[f"#neighbor{n_neighbor}"] = {
                "path" : sqrt_path,
                "center_indices" : target_centers,
                "neighbor_indices" : target_neighbors,
                "prec_types" : prec_types,
            }
                
            print(f"already exist: {sqrt_path}")
        else:
            path_forMapping.append((precision_path, sqrt_path))
    
    # Loop for all targets
    for precision_path, sqrt_path in path_forMapping:
        save_path_info = Path(sqrt_path)
        file_name = save_path_info.stem
        extension = save_path_info.suffix
        
        # Load precision matrices
        precision_dataSet = np.load(precision_path)
        centers = precision_dataSet["centers"]
        neighbors = precision_dataSet["neighbors"]
        sl_precisions = precision_dataSet["precision"]
        prec_types = precision_dataSet["prec_types"]
        print(f"load {precision_path}")
        
        # Reconstruct precision matrix from 1D matrix into 2D matrix
        _, n_neighbor = neighbors.shape
        n_center, n_source, n_element = sl_precisions.shape
        r, c = np.triu_indices(n_neighbor, k = 0) # get upper triangle indices from #neighbor x #neighbor matrix
        
        # Loop across all centers
        for chunk_i in trange(0, math.ceil(n_center / n_batch)):
            selecting = range(chunk_i * n_batch, min((chunk_i+1) * n_batch, n_center))
            
            target_centers = centers[selecting]
            n_target_center = len(target_centers)
            target_neighbors = neighbors[selecting]
            target_sl_precisions = sl_precisions[selecting]
            reconstructionMat = reconstruct_sl_precisionMats(target_sl_precisions, n_neighbor)
            
            # Calculate square root about the precision matrix 
            sqrt_precision = calc_sqrtMat(reconstructionMat) # shape: (#center * #source, #n_neighbor, #n_neighbor)
            sqrt_precision = sqrt_precision[:, r, c] # shape: (#center * #source, #comb(n_neighbor, 2))
            sqrt_precision = sqrt_precision.reshape((n_target_center, n_source, -1)) # shape: (#center, #source, #comb(n_neighbor, 2))
            sqrt_precision = sqrt_precision.astype(dtype)
        
            # Save the sqrt precision result
            chunk_save_path = os.path.join(save_dir_path, file_name + f"_chunk{chunk_i}" + extension)
            np.savez(chunk_save_path, 
                     centers = target_centers, 
                     neighbors = target_neighbors,
                     sqrt_precision = sqrt_precision,
                     prec_types = prec_types)
            print(f"save_path: {chunk_save_path}")
            
            del sqrt_precision
        del precision_dataSet
        del sl_precisions
    
        # Load all chunk datas and concatenate all datas
        chunk_paths = glob(os.path.join(save_dir_path, f"{file_name}_chunk*"))
        chunk_paths = sorted(chunk_paths, key = lambda x: int(re.search(r'chunk(\d+)', x).group(1)))
        chunkSets = [np.load(path) for path in chunk_paths]
        is_same_sources = np.all([chunkSets[0]["prec_types"] == chunkSet["prec_types"] for chunkSet in chunkSets])
        if is_same_sources:
            centers_ = np.concatenate([chunkSet["centers"] for chunkSet in chunkSets])
            neighbors_ = np.concatenate([chunkSet["neighbors"] for chunkSet in chunkSets])
        
            sqrt_precisions = []
            for chunkSet in chunkSets:
                sqrt_precisions.append(chunkSet["sqrt_precision"])
            sqrt_precision = np.concatenate(sqrt_precisions, axis = 0)
            del sqrt_precisions
        
            assert np.all([chunkSet["prec_types"] == chunkSets[0]["prec_types"] for chunkSet in chunkSets]), "Check"
            prec_types = chunkSets[0]["prec_types"]
        
            np.savez(sqrt_path, 
                     centers = centers_, 
                     neighbors = neighbors_,
                     sqrt_precision = sqrt_precision,
                     prec_types = prec_types)
            print(f"save: {sqrt_path}")
            del sqrt_precision
            
            result_info[f"#neighbor{n_neighbor}"] = {
                "path" : sqrt_path,
                "center_indices" : centers_,
                "neighbor_indices" : neighbors_,
                "prec_types" : prec_types,
            }
            
            for path in chunk_paths:
                os.system(f"rm {path}")
                print(f"remove: {path}")
    return result_info

def calc_sl_rdm_crossnobises(n_split_data: int,
                             unique_n_neighbors: np.array, 
                             precision_dir_path: str, 
                             measurements: np.array, 
                             masking_indexes: np.array, 
                             conditions: np.array, 
                             sessions: np.array,
                             n_thread_per_block: int = 1024,
                             dtype = np.float32):
    """
    Calculate searchlight crossnobis rdm on each center.
    However, this function calculate RDM differently if n_neighbor is different
    
    1. load precision matrix for corresponding specific #neighbor on precision_dir_path 
    2. Calculate RDM with searchlight way

    Notice) 
    This function load precision matrix on precision_dir_path.
    So you have to check whether the precision matrices are not overlapped.
    
    :param n_split_data: how many datas to process at once
    :param unique_n_neighbors: unique #neighbor 
    :param precision_dir_path: path saved for precision matrix
    :param measurements(shape: #data, #channel): measurment values
    :param masking_indexes(shape: #channel):  index of masking brain
    :param conditions(shape: #data): condition array corresponding to measurements
    :param sessions(shape: #data): session corressponding to conds
    :param n_thread_per_block(int): block per thread
    :param dtype: data type for storing array
    
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
                                                                n_thread_per_block = n_thread_per_block,
                                                                dtype = dtype)
        rdm_crossnobis_gpus = np.concatenate(rdm_crossnobis_gpus)
    
        # Make sl_rdms
        rdm_crossnobis_gpus = RDMs(rdm_crossnobis_gpus,
                                       pattern_descriptors = {
                                           "index" : np.arange(0, len(rdm_conds)).tolist(),
                                           "events" : rdm_conds.tolist(),
                                       },
                                       rdm_descriptors = {
                                           "voxel_index" : target_centers.tolist(),
                                           "index" : np.arange(0, len(target_centers)).tolist()
                                       })
        rdm_crossnobis_gpus.dissimilarity_measure = "crossnobis"

        # Acc
        rdms.append(rdm_crossnobis_gpus)
        centers.append(target_centers)
    
    # Concat
    centers = np.concatenate(centers)
    
    rdms = concat(rdms)
    rdms.rdm_descriptors["voxel_index"] = centers.tolist()
    
    # Reorder
    rdms = rdms.subsample(by = "voxel_index", value = np.sort(centers))
    
    return rdms

def calc_sl_rdm_crossnobis_SS(measurements: np.ndarray,
                              conditions: np.ndarray,
                              sessions: np.ndarray,
                              subSessions: np.ndarray,
                              centers: np.ndarray,
                              neighbors: np.ndarray,
                              sqrt_precisions: np.ndarray,
                              precision_types: np.ndarray,
                              masking_indexes: np.ndarray,
                              n_split_data: int,
                              n_thread_per_block: int = 1024,
                              dtype = np.float32):
    """
    Calculate searchlight crossnobis distance considering session and subSession
    
    :param measurements(shape: (#data, #channel)): measurment values
    :param conditions(shape: #data): condition per measurement
    :param sessions(shape: #data): session per measurement
    :param subSessions(shape: #data): subSession per measurement
    :param centers(shape: #center): center voxel index
    :param neighbors(#center, #neighbor): neighbor indices per center voxel index 
    :param sqrt_precisions(shape: (#channel, #precision_types, #precision_mat_element)): precision matrix per channel
    :param precision_types(shape: #precision_types): precision type corresponding to sqrt_precision elements
    :param masking_indexes(shape: #channel): index of masking brain
    :param n_split_data: how many datas to process at once
    :param n_thread_per_block: block per thread
    :param dtype: data type for storing array
    """
    if measurements.dtype != dtype:
        measurements = measurements.astype(dtype)
        sqrt_precisions = sqrt_precisions.astype(dtype)

    # Convert data types
    sessions = np.array(sessions).astype(str)
    subSessions = np.array(subSessions).astype(str)
    conditions = np.array(conditions).astype(str)
    
    # Get unique element according to appearance order
    uq_sessions = np.array(list(dict.fromkeys(sessions)))
    uq_subSessions = np.array(list(dict.fromkeys(subSessions)))
    uq_conditions = np.array(list(dict.fromkeys(conditions)))

    condition_index_info = {}
    for i in range(len(uq_conditions)):
        condition_index_info[uq_conditions[i]] = i
    
    session_index_info = {}
    for i in range(len(uq_sessions)):
        session_index_info[uq_sessions[i]] = i
        
    subSession_index_info = {}
    for i in range(len(uq_subSessions)):
        subSession_index_info[uq_subSessions[i]] = i
    
    # Reorder measurements & conditions
    measurement_sort_info = pd.DataFrame({
        "condition" : conditions,
        "session" : sessions,
        "subSession" : subSessions,
    })
    measurement_sort_info["origin_i"] = measurement_sort_info.index
    measurement_sort_info["condition_i"] = [condition_index_info[e] for e in measurement_sort_info["condition"]]
    measurement_sort_info["session_i"] = [session_index_info[e] for e in measurement_sort_info["session"]]
    measurement_sort_info["subSession_i"] = [subSession_index_info[e] for e in measurement_sort_info["subSession"]]
    measurement_sort_info = measurement_sort_info.sort_values(
        by=["session_i", "condition_i", "subSession_i"],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    measurements = measurements[measurement_sort_info["origin_i"]]

    # Reorder precisions
    precision_type_df = pd.DataFrame([e.split("-") for e in precision_types])
    precision_type_df.columns = ["session", "subSession"]
    precision_type_df["session_i"] = precision_type_df["session"].map(session_index_info)
    precision_type_df["subSession_i"] = precision_type_df["subSession"].map(subSession_index_info)
    precision_type_df["origin"] = precision_type_df.index
    precision_type_df = precision_type_df.sort_values(
        by=["session_i", "subSession_i"],
        ascending=[True, True]
    ).reset_index(drop=True)
    sqrt_precisions = sqrt_precisions[:, precision_type_df["origin"], :]

    # A number of data
    n_data, n_channel = measurements.shape
    n_precision = len(precision_types)
    n_uq_session = len(uq_sessions)
    n_uq_subSession = len(uq_subSessions)
    n_uq_condition = len(uq_conditions)
    n_center, n_neighbor = neighbors.shape
    n_precision_element = int(((n_neighbor * n_neighbor) - n_neighbor) / 2 + n_neighbor)
    
    # index information for calculating difference
    n_dissim = len(list(combinations(uq_conditions, 2)))
    diff_conditions = np.array(list(combinations(uq_conditions, 2)))
    diff_info_array = np.array([np.c_[np.repeat(session, n_dissim), diff_conditions] for session in uq_sessions])
    diff_info_df = pd.DataFrame(np.concatenate(diff_info_array, axis = 0))
    diff_info_df.columns = ["session", "cond1", "cond2"]
    diff_info_df["session_i"] = [session_index_info[session] for session in diff_info_df["session"]] # session index
    diff_info_df["uq_cond1_i"] = [condition_index_info[cond] for cond in diff_info_df["cond1"]] # cond1 index
    diff_info_df["uq_cond2_i"] = [condition_index_info[cond] for cond in diff_info_df["cond2"]] # cond2 index
    diff_info_df["cond1_i"] = diff_info_df["session_i"] * n_uq_condition + diff_info_df["uq_cond1_i"] # measurement index information of cond1
    diff_info_df["cond2_i"] = diff_info_df["session_i"] * n_uq_condition + diff_info_df["uq_cond2_i"] # measurement index information of cond2
    diff_info_df["dissim_i"] = np.tile(np.arange(n_dissim), n_uq_session) # dissim index 
    del diff_info_array

    # Prepare multiplication info for denoising
    mul_mapping = np.zeros((n_neighbor, n_neighbor))
    r, c = np.triu_indices(n_neighbor, k = 0)
    mul_mapping[r, c] = np.arange(r.shape[0])
    mul_mapping += mul_mapping.T
    idx = np.diag_indices(mul_mapping.shape[0])
    mul_mapping[idx] = mul_mapping[idx] / 2
    mul_mapping = mul_mapping.astype(int)
    mul_mapping = cuda.to_device(mul_mapping)

    # Prepare cross-validation info within same dissimilarity
    n_cv = len(list(combinations(uq_sessions, 2)))
    cv_diff_indices = np.zeros((n_split_data * n_cv * n_dissim, 4), dtype = np.uint32) 
    i = 0
    for center_i, dissim_i in product(np.arange(n_split_data), np.arange(n_dissim)):
        indices = np.where(diff_info_df["dissim_i"] == dissim_i)[0] + (center_i * n_uq_session * n_dissim)
        comb = np.array(list(combinations(indices, 2)))
    
        for j in range(len(comb)):
            diff1_i, diff2_i = comb[j]
            cv_diff_indices[i] = np.r_[center_i, dissim_i, diff1_i, diff2_i]
            i += 1
    cv_diff_df = pd.DataFrame(cv_diff_indices)
    cv_diff_df.columns = ["center_i", "dissim_i", "diff1_i", "diff2_i"]
    cv_diff_df['cv_i'] = cv_diff_df.groupby(['center_i', 'dissim_i']).cumcount()

    # Memory pool
    mempool = cp.get_default_memory_pool()
    n_block = int(np.ceil(n_split_data / n_thread_per_block))
    
    # Calculation
    rdm_outs = []
    for i in trange(0, n_center, n_split_data):
        """
        Select centers neighbors and square root precision matrix
        """
        # Select neighbors
        target_range = range(i, min(len(centers), i + n_split_data))
        target_centers = centers[target_range]
        target_neighbors = neighbors[target_range, :]
        
        n_target_center  = len(target_centers)
        target_sqrt_precisions = sqrt_precisions[target_range]
        target_sqrt_precisions = target_sqrt_precisions.reshape(n_target_center * n_precision, n_precision_element)
        target_sqrt_precisions = cuda.to_device(target_sqrt_precisions)
    
        """
        Apply mask
    
        Required GPU memory capacitiy
            - mask_out: (#center, #channel)
            - masked_measurements: (#center * #data, #neighbor)
        """
        # Masking
        mask = set_mask_cpu(target_neighbors, masking_indexes)
        
        ## Make mask
        masked_measurements = []
        for j in range(n_target_center):
            masked_measurements.append(measurements[:, mask[j] == 1])
        masked_measurements = np.array(masked_measurements)
    
        ## Apply mask to measurements
        masked_measurements = masked_measurements.reshape(n_target_center * n_data, n_neighbor)
        masked_measurements = cuda.to_device(masked_measurements)
        n_measurement, _ = masked_measurements.shape
        
        """
        Denoise
        
        Required GPU memory capacitiy
            - measurements: (#center * #data, #neighbor)
            - target_sqrt_precisions: (#center * #source, comb(#neighbor, 2))
            - prec_indices: (#center * #data)
            - denoised_masked_measurements: (#target_center * #data, #neighbor)
        """
        ## Prepare GPU kernel info
        if n_measurement % n_thread_per_block == 0:
            n_block_forDenoise = n_measurement // n_thread_per_block
        else:
            n_block_forDenoise = n_measurement // n_thread_per_block + 1
        
        ## Prepare information per measurement for denoising 
        measusrement_info_df = pd.concat([measurement_sort_info] * n_target_center, ignore_index=True)
        measusrement_info_df["target_center_i"] = np.repeat(np.arange(len(target_centers)), n_data)
        measusrement_info_df = measusrement_info_df[["target_center_i", "session", "subSession", "condition"]]
        measusrement_info_df["prec_i"] = measusrement_info_df["target_center_i"] * n_precision + measusrement_info_df["session"].map(session_index_info) * n_uq_subSession + measusrement_info_df["subSession"].map(subSession_index_info)
        prec_indices = cuda.to_device(measusrement_info_df["prec_i"].to_numpy())
        
        ## Denoise measurements
        denoised_masked_measurements = cuda.to_device(np.zeros(masked_measurements.shape))
        denoise[n_block_forDenoise, n_thread_per_block](masked_measurements, 
                                                        target_sqrt_precisions, 
                                                        prec_indices, 
                                                        mul_mapping, 
                                                        denoised_masked_measurements)
        cuda.synchronize()
    
        del masked_measurements
        del target_sqrt_precisions
        del prec_indices
        cuda.defer_cleanup()
    
        """
        Average data within same condition per session
        """
        # Skip...
        
        """
        Calculate difference between measurement on same session
    
        Required GPU memory capacitiy
            - measurements: (#center * #data, #neighbor)
            - diff_index_arrays: (#center * #data, #neighbor)
            - diff_index_arrays: (#center * #session #dissim, 2)
            - diff_measurements: (#center * #session #dissim, #neighbor)
        """
        ## Prepare GPU kernel info
        n_cal_diff = n_target_center * n_uq_session * n_dissim
        if n_cal_diff % n_thread_per_block == 0:
            n_block_forDiff = n_cal_diff // n_thread_per_block
        else:
            n_block_forDiff = n_cal_diff // n_thread_per_block + 1
    
        ## Prepare target index array to be differentiated
        target_center_indices = np.repeat(np.arange(n_target_center, dtype = np.uint32), n_uq_session * n_dissim)
        diff_measurement_index_df = pd.concat([diff_info_df[["cond1", "cond2", "cond1_i", "cond2_i"]]] * n_target_center, ignore_index=True) # repeat difference info up to cover all centers
        diff_measurement_index_df["center_i"] = target_center_indices
        diff_measurement_index_df["measurement_cond1_i"] = diff_measurement_index_df["center_i"] * n_data + diff_measurement_index_df["cond1_i"]
        diff_measurement_index_df["measurement_cond2_i"] = diff_measurement_index_df["center_i"] * n_data + diff_measurement_index_df["cond2_i"]
        diff_index_arrays = diff_measurement_index_df[["measurement_cond1_i", "measurement_cond2_i"]].to_numpy(dtype = np.uint32)
        diff_index_arrays = cuda.to_device(diff_index_arrays)
        del target_center_indices
    
        ## Calculate difference among conditions within same session
        diff_measurements = cuda.to_device(np.zeros((n_target_center * n_uq_session * n_dissim, n_neighbor), dtype = dtype))
        differentiate_measurements[n_block_forDiff, n_thread_per_block](denoised_masked_measurements, diff_index_arrays, diff_measurements)
        cuda.synchronize()
    
        del diff_index_arrays
        del denoised_masked_measurements
        cuda.defer_cleanup()
    
        """
        Cross-validated distance
        """
        target_cv_diff_indices = cuda.to_device(cv_diff_df[cv_diff_df["center_i"] < len(target_centers)].to_numpy(dtype = np.uint32))
        n_cal = target_cv_diff_indices.shape[0]
        
        if n_cal_diff % n_thread_per_block == 0:
            n_block_forCV = n_cal // n_thread_per_block
        else:
            n_block_forCV = n_cal // n_thread_per_block + 1
    
        distances_gpu = cuda.to_device(np.zeros((n_target_center, n_dissim, n_cv), dtype = dtype))
        calc_cv_distance[n_block_forCV, n_thread_per_block](diff_measurements, target_cv_diff_indices, distances_gpu, n_dissim)
        cuda.synchronize()
        distances = distances_gpu.copy_to_host()
        del distances_gpu
        del target_cv_diff_indices
        del diff_measurements
    
        distances = cp.mean(distances, axis = 2)
        distances = distances / n_neighbor
        rdm_outs.append(distances)
        
        cuda.defer_cleanup()
    return rdm_outs, uq_conditions

def calc_sl_rdm_crossnobises_SS(n_split_data: int,
                                unique_n_neighbors: np.array, 
                                precision_dir_path: str, 
                                masking_indexes: np.array, 
                                measurements: np.array, 
                                conditions: np.array, 
                                sessions: np.array,
                                subSessions: np.array,
                                n_thread_per_block: int = 1024,
                                dtype = np.float32):
    """
    Calculate searchlight crossnobis rdm on each center considering subSession.
    However, this function calculate RDM differently if n_neighbor is different
    
    1. load precision matrix for corresponding specific #neighbor on sqrt_precision_dir_path 
    2. Calculate RDM with searchlight way

    Notice) 
    This function load precision matrix on sqrt_precision_dir_path.
    So you have to check whether the precision matrices are not overlapped.
    
    :param n_split_data: how many datas to process at once
    :param unique_n_neighbors: unique #neighbor 
    :param sqrt_precision_dir_path: path saved for sqrt precision matrix
    :param measurements(shape: #data, #channel): measurment values
    :param masking_indexes(shape: #channel):  index of masking brain
    :param conditions(shape: #data): condition array corresponding to measurements
    :param sessions(shape: #data): session corressponding to conds
    :param n_thread_per_block(int): block per thread
    :param dtype: data type for storing array
    
    return (rsatoolbox.rdm.RDMs): RDM matrix about each brain coordinate
    """

    centers = []
    rdms = []
    for n_neighbor in unique_n_neighbors:
        # Load precision matrix
        sqrt_precision_dataSet_path = os.path.join(precision_dir_path, f"sqrt_precision_neighbor{n_neighbor}.npz")
        sqrt_precision_dataSet = np.load(sqrt_precision_dataSet_path)
        target_centers = sqrt_precision_dataSet["centers"]
        target_neighbors = sqrt_precision_dataSet["neighbors"]
        sqrt_precisions = sqrt_precision_dataSet["sqrt_precision"]
        prec_types = sqrt_precision_dataSet["prec_types"]
        print(f"loaded: {sqrt_precision_dataSet_path}")
        
        # Calculate RDM with searchlight way
        rdm, rdm_conds = calc_sl_rdm_crossnobis_SS(measurements = measurements,
                                                    conditions = conditions,
                                                    sessions = sessions,
                                                    subSessions = subSessions,
                                                    centers = target_centers,
                                                    neighbors = target_neighbors,
                                                    sqrt_precisions = sqrt_precisions,
                                                    precision_types = prec_types,
                                                    masking_indexes = masking_indexes,
                                                    n_split_data = n_split_data)
        rdm_arrays = np.concatenate(rdm, axis = 0)
    
        # Make sl_rdms
        rdm_crossnobis_gpus = RDMs(rdm_arrays,
                                   pattern_descriptors = {
                                       "index" : np.arange(0, len(rdm_conds)).tolist(),
                                       "events" : rdm_conds.tolist(),
                                   },
                                   rdm_descriptors = {
                                       "voxel_index" : target_centers.tolist(),
                                       "index" : np.arange(0, len(target_centers)).tolist()
                                   })
        rdm_crossnobis_gpus.dissimilarity_measure = "crossnobis"

        # Acc
        rdms.append(rdm_crossnobis_gpus)
        centers.append(target_centers)
        
    # Concat
    centers = np.concatenate(centers)
    rdms = concat(rdms)
    rdms.rdm_descriptors["voxel_index"] = centers.tolist()
    
    # Sorting
    sorted_voxel_indices = np.argsort(rdms.rdm_descriptors["voxel_index"])
    rdms.dissimilarities = rdms.dissimilarities[sorted_voxel_indices]
    rdms.rdm_descriptors["voxel_index"] = centers[sorted_voxel_indices]

    return rdms
    