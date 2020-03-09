# Paper: https://www.aclweb.org/anthology/D15-1228.pdf

import numpy as np
from typing import Callable


def distance_to_subspace(v: np.array, S: np.array) -> float:
    '''Calculates a distance from vector to subspace'''
    return np.linalg.norm(v - (v @ S.T) @ S)


def max_semantic_volume_summary(R: np.array, L: int, dist: Callable) -> list:
    '''Implementation of the "Extractive Summarization by Maximizing Semantic Volume".
    
    Note: The budget constraint is number of sentences in summary, not the number of words.
    
    :param R:    Sentence reprezentations
    :param L:    Number of sentences in summary
    :param dist: Function that computes distance to subspace
    
    :return: Sentence indices
    '''
    N = R.shape[0]

    B = np.ndarray(shape=(0,R.shape[1]))
    idx = []

    c = R.sum(axis=0) / N

    p = np.argmax(np.linalg.norm(R - c, axis=1))
    idx.append(p)

    q = np.argmax(np.linalg.norm(R - R[p], axis=1))
    idx.append(q)

    b_0 = R[q] / np.linalg.norm(R[q])
    B = np.append(B, b_0.reshape(1,-1), axis=0)

    for i in range(L-2):
        d = np.apply_along_axis(dist, 1, R, B)
        r = np.argmax(np.ma.masked_array(d, mask=np.in1d(range(N),idx))) # mask the indices already chosen
        idx.append(r)

        b_r = R[r] / np.linalg.norm(R[r])
        B = np.append(B, b_r.reshape(1,-1), axis=0)
        
    return idx