
import os
import sys
from numba import cuda, jit

if os.getenv("boostrsa_isRunSource"):
    sys.path.append(os.getenv("boostrsa_source_home"))
    from cores.gpu.basic_operations import matmul
else:
    from boostrsa.cores.gpu.basic_operations import matmul

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
def rdm_from_kernel(kernels, div, out):
    """
    Calculate rdm matrix
    
    :param kernels(Device array): kernel, shape: (n_data, n_fold, n_cond, n_cond))
    :param div(int): div value
    :param out(Device array): rdm output, shape: (n_data, n_fold, n_dissim)
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
def calc_kernel(measurments, precisions, fold_info, out1, out2):
    """
    Calculate rdm kernel for calculating crossnobis
    
    :param measurments(Device array): , shape: (n_data, n_run, n_cond, n_neighbor)
    :param precisions(Device array): , shape: (n_data, n_fold, n_neighbor, n_neighbor)
    :param fold_info(Device array): fold information - [[fold1, fold2], ...]
    :param out1(Device array): intermediate matmul output , shape: (n_data, n_fold, n_cond, n_neighbor) 
    :param out2(Device array): kernel output , shape: (n_data, n_fold, n_cond, n_cond))
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

