import numpy as np

class PCA:

    def __init__(self, percent_precision):
        self.precision = percent_precision/100
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, functions needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # calculate variance of eigenvectors
        variancevectors = np.var(eigenvectors, axis=0)
        sumvar = np.sum(variancevectors)
        varto = 0
        precision = 0
        i = 0
        while (precision < self.precision):
            varto += variancevectors[i]
            precision = varto/sumvar
            i += 1
        
        n_components = i
        
        self.components = eigenvectors[:n_components]


    def transform(self, X):
        # projects data
        X = X - self.mean
        return np.dot(X, self.components.T)