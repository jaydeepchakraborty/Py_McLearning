import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

A =  np.array([[0, 1, 0, 0, 1], [0, 0, 1, 1, 1],[1, 1, 0, 1, 0]])
A_sparse = sparse.csr_matrix(A)

similarities = cosine_similarity(A_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

#also can output sparse matrices
similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


'''

Results:

pairwise dense output:
[[ 1.          0.40824829  0.40824829]
[ 0.40824829  1.          0.33333333]
[ 0.40824829  0.33333333  1.        ]]

pairwise sparse output:
(0, 1)  0.408248290464
(0, 2)  0.408248290464
(0, 0)  1.0
(1, 0)  0.408248290464
(1, 2)  0.333333333333
(1, 1)  1.0
(2, 1)  0.333333333333
(2, 0)  0.408248290464
(2, 2)  1.0

'''