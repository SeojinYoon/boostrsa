
import os
import sys
import cupy as cp
import numpy as np
from numba import cuda, jit
from numba.cuda.cudadrv.devicearray import DeviceNDArray

if os.getenv("boostrsa_isRunSource"):
    sys.path.append(os.getenv("boostrsa_source_home"))
    from cores.gpu.basic_operations import matmul, matmul_upperTmat, minus, dot
else:
    from boostrsa.cores.gpu.basic_operations import matmul, matmul_upperTmat, minus, dot

@jit(nopython=True)
def upper_tri_1d_index(i, j, n_col, k):
    """
    Get upper triangle 1d index
    
    if k = 1)
    
    (0,1), (0,2), (0,3), (0,4) -> 0, 1, 2, 3
           (1,2), (1,3), (1,4) -> 4, 5, 6
                  (2,3), (2,4) -> 7, 8
                         (3,3) -> 9
                         
    :param i: row index
    :param j: column index
    :param n_col: column number
    :param k: #padding
    """
    if i > j:
        return None
    else:
        sum_val = 0
        for loop_row_i in range(0, i):
            sum_val += (n_col - k) # maximum filled count of row.
            sum_val += (-1) * loop_row_i # non-filled element is increased as row value is increased.
        return sum_val + (j - i - k)

@jit(nopython=True)
def lower_tri_1d_index(i, j):
    """
    Get lower triangle 1d index
    
    :param i: row index
    :param j: column index
    """
    
    if i < j:
        return None
    else:        
        total_fill = 0
        for pr_row_i in range(1, i + 1):
            total_fill += (pr_row_i - 1)
        return total_fill + j

@cuda.jit
def diag(matrices, out):
    i = cuda.grid(1)

    if i < len(matrices):
        matrix = matrices[i]

        n_row = len(matrix)
        for j in range(n_row):
            out[i][j] = matrix[j][j]

@cuda.jit
def eyes(out):
    i = cuda.grid(1)

    nd = out.shape[0]
    nr = out.shape[1]
    nc = out.shape[2]

    if i < len(out):
        for j in range(nr):
            out[i][j][j] = 1

@cuda.jit
def rdm_from_kernel(kernels: DeviceNDArray, 
                    div: int, 
                    out: DeviceNDArray):
    """
    Calculate rdm matrix
    
    :param kernels(shape: (#data, #fold, #cond, #cond)): kernel
    :param div: denominator for each output element
    :param out(shape: (#data, #fold, #dissim)): rdm output
    """
    n_data = kernels.shape[0]
    n_validation = kernels.shape[1]
    n_cond = kernels.shape[-1]
    
    i, j = cuda.grid(2)
    
    if i < n_data:
        if j < n_validation:
            kernel = kernels[i][j]
            
            for row_i in range(n_cond):
                for column_i in range(n_cond):
                    if row_i < column_i:
                        dissim_i = int(upper_tri_1d_index(row_i, column_i, n_cond, 1))

                        # Assign dissim value
                        v1 = kernel[row_i][row_i] + kernel[column_i][column_i]
                        v2 = kernel[row_i][column_i] + kernel[column_i][row_i]
                        out[i][j][dissim_i] = (v1 - v2) / div

@cuda.jit
def calc_kernel(measurments: DeviceNDArray, 
                precisions: DeviceNDArray, 
                fold_info: DeviceNDArray, 
                out1: DeviceNDArray, 
                out2: DeviceNDArray):
    """
    Calculate rdm kernel for calculating crossnobis
    
    :param measurments(shape: (#data, #run, #cond, #neighbor)):
    :param precisions(shape: (#data, #fold, #neighbor, #neighbor)): 
    :param fold_info: fold information - [[fold1, fold2], ...]
    :param out1(shape: (#data, #fold, #cond, #neighbor)): intermediate matmul output
    :param out2(shape: (#data, #fold, #cond, #cond)): kernel output
    """
    n_data = out1.shape[0]
    n_validation = out1.shape[1]

    i, j = cuda.grid(2)
    if i < n_data:
        if j < n_validation:
            data1_i, data2_i = fold_info[j]
            
            # measurements1 @ noise @ measurements2.T
            matmul(measurments[i][data1_i], precisions[i][j], out1[i][j])
            matmul(out1[i][j], measurments[i][data2_i].T, out2[i][j])

@cuda.jit
def denoise(measurements: DeviceNDArray,
            sqrt_precisions: DeviceNDArray,
            sqrt_precisionMat_indices: DeviceNDArray,
            mul_mapping : DeviceNDArray,
            output: DeviceNDArray):
    """
    Denoise measurement using precision matrix

    denoised_measurement = measurement @ (precision^(1/2))
    
    :param measurements(shape: #measurement, n_neighbor): measurement arrays
    :param sqrt_precisions(shape: (#center * #source, #element)): precision matricies
    :param sqrt_precisionMat_indices(shape: #measurement): corresponding precision mat index per measurement
    :param mul_mapping(shape: (#element, #element)): index mapping to be multiplied
    :param output(shape: (#measurement, #n_neighbor)): output array
    """
    i = cuda.grid(1)
    if i >= len(measurements):
        return
        
    sqrt_precisionMat_i = sqrt_precisionMat_indices[i]
    matmul_upperTmat(measurements[i], sqrt_precisions[sqrt_precisionMat_i], mul_mapping, output[i])

@cuda.jit
def differentiate_measurements(measurements, diff_index_arrays, output):
    i = cuda.grid(1)
    if i >= len(diff_index_arrays):
        return
    
    cond1_i, cond2_i = diff_index_arrays[i]
    minus(measurements[cond1_i], measurements[cond2_i], output[i])

@cuda.jit
def calc_cv_distance(diff_measurements, cv_diff_df, output, n_dissim):
    i = cuda.grid(1)
    if i >= len(cv_diff_df):
        return
    center_i, dissim_i, diff1_i, diff2_i, cv_i = cv_diff_df[i]
    dot(diff_measurements[diff1_i], diff_measurements[diff2_i], output[center_i][dissim_i], cv_i)

def calc_sqrtMat(matrices: np.ndarray) -> np.ndarray:
    """
    Calculate sqrt using eigen value

    Q @ diag(sqrt(w)) @ Q^T
    
    :param matrices(shape - #batch, #element, #element): A matrix for which you want to take the square root

    return (shape - #batch, #element, #element)
    """
    # Memory pool
    mempool = cp.get_default_memory_pool()
    
    # Eigen value decomposition
    w, Q = cp.linalg.eigh(matrices) # w:(n_batch, n_element), Q:(n_batch, n_element, n_element)
    w = cp.clip(w, 0, None)
    
    # Calculate sqrt matrix
    w_sqrt = cp.sqrt(w)
    X = (Q * w_sqrt[..., None, :]) @ cp.swapaxes(Q, -1, -2)   # (b,n,n)
    X = cp.asnumpy(X)

    # Clean memory
    mempool.free_all_blocks()
    
    return X
    
if __name__ == "__main__":
    # diag
    matrices = np.array([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        [[9, 8, 7],
         [6, 5, 4],
         [3, 2, 1]],
        [[2, 0, 1],
         [0, 3, 0],
         [4, 0, 5]]
    ], dtype=np.float32)
    
    n_matrices = matrices.shape[0]
    n_rows = matrices.shape[1]
    out = np.zeros((n_matrices, n_rows), dtype=np.float32)
    
    d_matrices = cuda.to_device(matrices)
    d_out = cuda.to_device(out)
    
    threads_per_block = 32
    blocks_per_grid = (n_matrices + (threads_per_block - 1)) // threads_per_block
    diag[blocks_per_grid, threads_per_block](d_matrices, d_out)