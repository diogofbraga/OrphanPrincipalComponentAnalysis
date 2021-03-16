#!/usr/bin/python

import numpy as np

def computeTanimotoMatrix_Diff(X1, X2):
    """
    calculates tanimoto kernel matrix for two different data matrices X1, X2
    (could also be realised via a global data matrix x and different index sets defining X1 and X2)
    """
    NUM = np.dot(X1, X2.T)
    X1X1T = np.diag(np.dot(X1,X1.T)).reshape((-1,1))
    X2X2T = np.diag(np.dot(X2,X2.T)).reshape((1,-1))
    DENOM = X1X1T + X2X2T - NUM
    Ktan = NUM/DENOM.astype(float)
    return Ktan

def computeTanimotoMatrix(X):
    """
    calculates tanimoto kernel matrix for data matrix X
    """
    NUM = np.dot(X, X.T)
    XXT_diag = np.diag(NUM)
    XXT_diag_column = XXT_diag.reshape((-1,1))
    XXT_diag_row = XXT_diag.reshape((1,-1))
    DENOM = XXT_diag_column + XXT_diag_row - NUM
    Ktan = NUM/DENOM.astype(float)
    return Ktan


def main():
    """
    example: one data matrix X (e.g. our global matrix containing x_1, ..., x_m)
    """
    X = np.array([[1,0,1,1],
                    [1,1,0,1],
                    [0,1,0,0]])
    K_Diff = computeTanimotoMatrix_Diff(X,X)
    K = computeTanimotoMatrix(X)
    print(K_Diff) #outputs should be the same!!
    print(K)

    """
    example: two data matrices (e.g. X1=global data matrix and X2=[x], could be used in the calculation formula for h_o(x))
    """
    X1 = np.array([[1,0,1,1],
                    [1,1,0,1],
                    [0,1,0,0]])
    X2 = np.array([[1,1,0,1]])
    K_Diff = computeTanimotoMatrix_Diff(X1,X2)
    print(K_Diff)

if __name__ == "__main__":
    main()





