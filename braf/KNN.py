# A module to evaluate KNN 
import numpy as np

def euclidean_distance(x0, x1):
    d2 = np.square(x0[None, :, :] - x1[:, None, :])
    d2 = np.sum(d2, axis=-1)
    return np.sqrt(d2)

class KNN:
    '''
    ref: https://m0nads.wordpress.com/tag/knn/ to implement this with concise codes
    KNN module
    '''
    def __init__(self):
        pass

    @staticmethod
    def get_k_neighbors(minor, major, k=5):
        '''
        Find k nearest neighbors
        Returns:
            indicies of k nearest neighbors: ndarray, NOT sorted
        '''
        distance = euclidean_distance(major, minor)
        k_nearest_partition = np.argpartition(distance, k, axis=-1)[:, :k]
        return k_nearest_partition