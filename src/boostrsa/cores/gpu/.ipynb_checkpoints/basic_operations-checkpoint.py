
from numba import cuda, jit

@cuda.jit
def calc_outerProduct(vec1, vec2, out):
    """
    Calculate outer product between vector1 and vector2.
    This is same as np.outer(vec1, vec2).
    
    :param vec1(np.array): vector1
    :param vec2(np.array): vector2
    :param out(cuda.cudadrv.devicearray.DeviceNDArray - shape: (#vec1_component, #vec2_component)): output array to store outer product result
    """
    i = cuda.grid(1)
    
    for j, e1 in enumerate(vec1):
        for k, e2 in enumerate(vec2):
            out[j][k] = e1 * e2

@cuda.jit
def outer_sum(matrices, out):
    """
    Calculate outer product and accumulating the result
    
    1. Calculate outer product to each data
        - np.outer(data, data): the data is same
    2. Accumulate outer result to output array iterating over all datas

    :param matrices(np.array - shape: (#run, #data, #channel)): measurement matrices
    :param out(cuda.cudadrv.devicearray.DeviceNDArray - shape: (shape: (#run, #channel, #channel))): output array to store data after calculation
    """
    i = cuda.grid(1)

    if i < len(matrices):
        matrix = matrices[i]

        for m_line in matrix:
            for j, e1 in enumerate(m_line):
                for k, e2 in enumerate(m_line):
                    out[i][j][k] += e1 * e2

@cuda.jit
def outer_sum_square(matrices, out):
    """
    Calculate outer product, square, and accumulating the result
    
    1. Calculate outer product to each data
        - np.outer(data, data): the data is same
    2. Calculate square product on the result
    3. Accumulate outer result to output array iterating over all datas

    :param matrices(np.array - shape: (#run, #data, #channel)): measurement matrices
    :param out(cuda.cudadrv.devicearray.DeviceNDArray - shape: (shape: (#run, #channel, #channel))): output array to store data after calculation
    """
    i = cuda.grid(1)

    if i < len(matrices):
        matrix = matrices[i]

        for m_line in matrix:
            for j, e1 in enumerate(m_line):
                for k, e2 in enumerate(m_line):
                    out[i][j][k] += (e1 * e2) ** 2

@cuda.jit
def scaling(out, lambs):
    i = cuda.grid(1)
    lamb = lambs[i]

    nd = out.shape[0]
    nr = out.shape[1]
    nc = out.shape[2]

    if i < len(out):
        for j in range(nr):
            for k in range(nc):
                if j != k:
                    out[i][j][k] = (1 - lamb)

@cuda.jit(device=True, inline=True)
def matmul(a,b, out):
    """
    Matrix multiplication a @ b
    
    :param a(np.array): 2d matrix
    :param b(np.array): 2d matrix
    :param out(device array): output
    """
    ar,ac = a.shape 
    br,bc = b.shape 
    
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                out[i,j] += a[i,k] * b[k,j]
    return out

if __name__ == "__main__":
    dummy_data = np.array([
        [
            [1,2,3], 
            [4,5,6],
            [5,6,7],
        ],
        [
            [7,8,9], 
            [0,1,2],
            [3,4,5],
        ],
    ])
    n_run, n_point, n_channel = dummy_data.shape
    calc_outerProduct[1,1](dummy_data[0][0], dummy_data[0][1], out)

    out_sum_device = cuda.to_device(np.zeros((n_run, n_channel, n_channel)))
    outer_sum[1,1](dummy_data, out_sum_device)

    out_sum_device = cuda.to_device(np.zeros((n_run, n_channel, n_channel)))
    outer_sum_square[1,1](dummy_data, out_sum_device)
    