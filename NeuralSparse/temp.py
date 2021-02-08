from scipy.sparse import csc_matrix
import numpy as np
from scipy.sparse.linalg import svds, eigs
from    utils import *

adj, features, all_labels, train, val, test = load_data('pubmed', task_type='semi')
# adj = adj.tocsc().astype(np.float32)


A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
u, s, vt = svds(A, k=2)
print(s)
