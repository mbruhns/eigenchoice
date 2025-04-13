import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA


class PCSelector:
    def __init__(self, data, pca):
        self.data = data
        self.pca = pca

    def _fit_transform(self, data):
        pass

    def _get_explained_variance(self):
        return self.pca.explained_variance_ratio_

    def _get_explained_variance_cum(self):
        return np.cumsum(self.pca.explained_variance_ratio_)

    def _get_n_components(self):
        return self.pca.n_components_
