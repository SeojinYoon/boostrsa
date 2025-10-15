
from numba import cuda, jit
from numba.cuda.cudadrv.devicearray import DeviceNDArray

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
def matmul(a: DeviceNDArray, 
           b: DeviceNDArray, 
           out: DeviceNDArray):
    """
    Matrix multiplication a @ b
    
    :param a(shape - 2d): 2d matrix
    :param b(shape - 2d): 2d matrix
    :param out(shape - 2d): output
    """
    ar,ac = a.shape 
    br,bc = b.shape 
    
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                out[i,j] += a[i,k] * b[k,j]

def matmul2(a: DeviceNDArray, 
            b: DeviceNDArray, 
            out: DeviceNDArray):
    """
    Matrix multiplication - vector @ array(2d)

    :param a(shape: 1d): vector
    :param b(shape: 2d): 2d array
    :param output(shape: 1d): output array
    """
    n_component_A = a.shape[0]
    n_row_B, n_col_B = b.shape

    for i in range(n_component_A):
        for j in range(n_row_B):
            out[i] += a[j] * b[j, i]

@cuda.jit(device=True, inline=True)
def matmul_upperTmat(vector: DeviceNDArray, 
                     upperTmat: DeviceNDArray, 
                     mul_mapping: DeviceNDArray, 
                     output: DeviceNDArray):
    """
    Multiply matrix with vector and symmetric matrix

    calculation: vector @ upperTmat
    
    :param vector(shape : #element): vector
    :param upperTmat(shape: #comb(#element, 2)): upper triangle matrix including diagnoal element
    :param mul_mapping(shape: (#element, #element)): index mapping to be multiplied
    :param output(shape: #element): output array
    """
    for vec_i in range(len(vector)):
        for row_i in range(len(mul_mapping)):
            output[vec_i] += vector[row_i] * upperTmat[mul_mapping[row_i, vec_i]]

@cuda.jit(device=True, inline=True)
def minus(a: DeviceNDArray, 
          b: DeviceNDArray, 
          out: DeviceNDArray):
    """
    Matrix multiplication a @ b
    
    :param a(shape - 1d): 1d matrix
    :param b(shape - 1d): 1d matrix
    :param out(shape - 1d): output
    """
    n_a = len(a)
    
    for i in range(n_a):
        out[i] = a[i] - b[i]

@cuda.jit(device=True, inline=True)
def dot(a1, a2, output, output_i):
    for i in range(len(a1)):
        output[output_i] += a1[i] * a2[i]
        
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

    # Matmul 1
    input_ = cuda.to_device(np.array([[1,2,3]]))
    array = cuda.to_device(np.arange(1, 10).reshape(3,3))
    result = cuda.to_device(np.zeros((3,3)).reshape(3,3))
    matmul(input_, array, result)
    
    # Matmul 2
    input_ = cuda.to_device(np.array([1,2,3]))
    array = cuda.to_device(np.arange(1, 10).reshape(3,3))
    result = cuda.to_device(np.zeros(3))
    matmul2(input_, array, result)

    # Matmul upper triangle mat
    vector = cuda.to_device(np.array([1,2,3]))
    n_element = vector.shape[0]
    r_, c_ = np.triu_indices(n_element, k = 0)
    upperTmat = np.array([
        [1,2,3],
        [2,4,5],
        [3,5,2],
    ])
    upperTmat = cuda.to_device(upperTmat[r_, c_])
    
    mul_mapping = np.zeros((n_element, n_element))
    mul_mapping[r_, c_] = np.arange(r_.shape[0])
    mul_mapping += mul_mapping.T
    idx = np.diag_indices(mul_mapping.shape[0])
    mul_mapping[idx] = mul_mapping[idx] / 2
    mul_mapping = mul_mapping.astype(int)
    mul_mapping = cuda.to_device(mul_mapping)
    output = cuda.to_device(np.zeros(n_element))

    matmul_upperTmat(vector = vector, 
                 upperTmat = upperTmat, 
                 mul_mapping = mul_mapping, 
                 output = output)

    # Minus
    a = cuda.to_device(np.ones(3))
    b = cuda.to_device(np.ones(3) * 2)
    output = cuda.to_device(np.zeros(3))
    
    @cuda.jit
    def minus_jit(a, b, out):
        minus(a, b, out)
    minus_jit[1,1](a, b, output)

    # Dot product
    a1 = np.array([1,2,3])
    a2 = np.array([1,2,3])
    output = np.array([0])
    
    @cuda.jit
    def dot_test(a1, a2, output):
        dot(a1, a2, output, 0)
    dot_test[1,1](a1, a2, output)