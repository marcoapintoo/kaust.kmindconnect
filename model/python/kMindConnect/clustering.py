import numpy as np
from .kmeans_helper import kmeans
import statsmodels.tsa.vector_ar.var_model as var_model

class OriginalKMeans:
    def __init__(self):
        self.cluster_centres = None
        self.clustered_coefficients = None
        self.expanded_time_series = None
        self.length_by_cluster = None
        self.time_varying_states_var_coefficients = None
    
    def _vns_clustering(self, y, K):
        min_dist = 1e100
        C = None
        for i in range(20):
            ci, S, dist = kmeans(y, centres=y[np.random.choice(np.arange(len(y)), K)],  metric="euclidean")
            if np.mean(dist) < min_dist:
                C = ci
                min_dist = np.mean(dist)
        C, _, _ = kmeans(y, centres=C, metric="chebyshev")
        C, _, _ = kmeans(y, centres=C, metric="cityblock")
        C, _, _ = kmeans(y, centres=C, metric="euclidean")
        C, _, _ = kmeans(y, centres=C, metric="chebyshev")
        C, _, _ = kmeans(y, centres=C, metric="cityblock")
        C, St_km, _ = kmeans(y, centres=C, metric="cityblock")
        return C, St_km
    
    def fit(self, time_coefficients, y):
        p = 1 #VAR model order
        K = 3 #Number of states
        wlen = 50 #Window length
        shift = 1 #Window shift
        min_r_len = 5 #Minimum length for each regime
        N, T = y.shape

        C, St_km = self._vns_clustering(time_coefficients.T, K)
        # Pooling samples for regimes
        St = np.zeros((N, T, K))
        tj = np.zeros(K, dtype="i")
        for j in range(K):
            t = 0
            for i in range(T):
                if St_km[i] == j:
                    St[:, t, j] = y[:, i]
                    t = t + 1
            tj[j] = t - 1

        # Estimate state-specific VAR
        A_km = np.zeros((N, N * p, K))
        for j in range(K):
            A_km[:, :, j] = var_model.VAR(St[:, :tj[j], j].T).fit(maxlags=p, method="ols", ic=None, trend="nc").params #Fit a VAR

        self.cluster_centres = C
        self.clustered_coefficients = St_km
        self.expanded_time_series = St
        self.length_by_cluster = tj
        self.time_varying_states_var_coefficients = A_km
