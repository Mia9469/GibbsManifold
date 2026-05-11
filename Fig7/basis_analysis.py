import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import FastICA
from sklearn.feature_selection import mutual_info_regression

def linear_independence(matrix):
    _, s, _ = svd(matrix)
    rank = np.sum(s > 1e-12)
    return rank == matrix.shape[1]

def orthogonality(func1, func2):
    inner_product = np.sum(func1 * func2)
    return np.isclose(inner_product, 0, atol=1e-12)

def mutual_information(basis_functions, data):
    # Perform ICA
    ica = FastICA(n_components=basis_functions.shape[1], random_state=42)
    coefficients = ica.fit_transform(data)
    
    # Calculate mutual information
    mi_matrix = np.zeros((basis_functions.shape[1], basis_functions.shape[1]))
    for i in range(basis_functions.shape[1]):
        for j in range(i, basis_functions.shape[1]):
            mi = mutual_info_regression(coefficients[:, i].reshape(-1, 1), coefficients[:, j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix

def decorrelation(matrix):
    corr = np.abs(np.corrcoef(matrix, rowvar=False))
    return np.ones(corr.shape)-corr

