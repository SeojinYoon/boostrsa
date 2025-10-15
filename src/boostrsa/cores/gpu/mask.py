
from numba import cuda, jit

@cuda.jit
def set_mask_gpu(neighbors, brain_1d_indexes, out):
    """
    Set neighbor mask(iterate over all neighbors)

    :param neighbors(np.array): list of neighbor , shape: (#center, #neighbor)
    :param brain_1d_indexes(np.array): 1d location converted from 3D brain coordinate (x,y,z) , shape: #channel
    :param out: masked_residual, output device memory , shape: (#center, #channel)
    """
    i = cuda.grid(1)

    if i < len(neighbors):
        neighbor_positions = neighbors[i]

        for neighbor_pos in neighbor_positions:
            for brain_i, brain_pos in enumerate(brain_1d_indexes):
                if brain_pos == neighbor_pos:
                    out[i][brain_i] = 1
if __name__ == "__main__":
    n_split_data = 2024
    n_channel = 200000
    n_thread_per_block = 1024
    
    n_block = int(np.ceil(n_split_data / n_thread_per_block))
    mask_out = cuda.to_device(np.zeros((n_split_data, n_channel)))
    target_neighbors = neighbors[0:0 + n_split_data, :]
    set_mask[n_block, n_thread_per_block](target_neighbors, masking_indexes, mask_out)