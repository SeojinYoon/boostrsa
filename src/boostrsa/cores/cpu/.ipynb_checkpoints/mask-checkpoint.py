
import numpy as np

def set_mask(neighbors, brain_1d_indexes):
    """
    Set neighbor mask(iterate over all neighbors)

    :param neighbors(np.array - shape: (#center, #neighbor)): list of neighbor
    :param brain_1d_indexes(np.array - shape: (#channel)): 1d location index converted from 3D brain coordinate (x,y,z)
    :param out: masked_residual(np.array - shape: (#center, #channel)): output device memory
    """
    return np.array([np.where(np.isin(mask_1d_indexes, target_neighbor), 1, 0) for target_neighbor in neighbors])
