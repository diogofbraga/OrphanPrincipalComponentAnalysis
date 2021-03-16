import numpy as np
from cvxopt import matrix, solvers

def svr_train(K, Xidx, y, C, eps):
    nu = 1/float(C)
    # K -> kernel matrix over labelled examples, K = original gram matrix + np.ones((n,n))
    # Xidx -> indices of the data rows in the original data object
    # y -> vector of labels (affinities), 1-dimensional np.array of length n
    # nu -> norm-regularisation parameter (C regularisation parameter of the error term)
    # eps -> margin parameter of epsilon-insensitive loss
    n = len(y) # number of training examples
    I_n = np.identity(n)
    I_nn = np.identity(2*n)
    H_1 = np.concatenate((I_n, -I_n), axis=1)
    H_2 = np.concatenate((I_n, I_n), axis=1)
    P = matrix(1/float(nu) * np.dot(np.dot(H_1.transpose(), K) , H_1))
    q = matrix((eps * np.sum(H_2, axis=0) - np.dot(y.reshape(1,-1), H_1)).reshape(-1))
    G = matrix(np.concatenate((I_nn, -I_nn), axis=0))
    h = matrix(np.concatenate((np.ones(2*n), np.zeros(2*n)), axis=0))
    solvers.options['show_progress'] = False
    duals = solvers.qp(P, q, G, h)['x'] # x = (\alpha_1,...,\alpha_n,\hat{\alpha}_1,...,\hat{\alpha}_n )^T
    pi = 1/float(nu) * np.dot(H_1, np.array(duals).reshape(-1)) # calculation of kernel expansion \pi via alpha-variables
    lim = 0.0001 # tolerance parameter, very small entries are assumed to be zero
    svs = []
    coeffs = []
    for i in range(n):
        if np.abs(pi[i]) > lim:
            svs.append(Xidx[i])
            coeffs.append(pi[i])
    # svs -> list of support vector indices
    # coeffs -> list of support vector coefficients (coefficient = result from the SVR)
    return np.array(svs), np.array(coeffs)


def predict(K_svs, coefs):
    return np.dot(K_svs, coefs)
