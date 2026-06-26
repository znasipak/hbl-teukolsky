#utils.py

import numpy as np

def chop_matrix(mat, tol = 1e-16, buffer = 0, copy=True):
    if copy:
        chopped_mat = mat.copy()
    else:
        chopped_mat = mat
    
    chopped_mat[np.abs(chopped_mat) < tol] = 0
    nonzero_count = buffer + np.max(np.count_nonzero(chopped_mat, axis = 1))
    chopped_mat = chopped_mat[:, :nonzero_count]

    return chopped_mat