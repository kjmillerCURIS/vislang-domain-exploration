import os
import sys
import numpy as np
from scipy.sparse import coo_array

def check_matrix_properties(M, N):
    M = int(M)
    N = int(N)

    v_list = np.concatenate((np.ones(2 * M * N + M), -np.ones(N)))
    i_list = np.concatenate((np.arange(M * N), np.arange(M * N), (M * N) * np.ones(M + N)))
    j_list = np.concatenate((np.tile(np.arange(M), N), np.repeat(np.arange(N) + M, M), np.arange(M + N)))
    A_sparse = coo_array((v_list, (i_list, j_list)), shape=(M * N + 1, M + N))
    A_dense = A_sparse.toarray()
    print('M = %d, N = %d'%(M, N))
    print('condition number = ' + str(np.linalg.cond(A_dense)))
    print('rank = ' + str(np.linalg.matrix_rank(A_dense)))
    print('singular_values = ' + str(np.linalg.svd(A_dense)[1]))
    import pdb
    pdb.set_trace()

def usage():
    print('Usage: python check_matrix_properties.py <M> <N>')

if __name__ == '__main__':
    check_matrix_properties(*(sys.argv[1:]))
