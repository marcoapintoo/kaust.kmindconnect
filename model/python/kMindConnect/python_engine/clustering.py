import numpy as np

from .kmeans_helper import kmeans
import statsmodels.tsa.vector_ar.var_model as var_model

class Clustering:
    def __init__(self, engine, var_order, number_states):
        self.engine = engine
        self.var_order = var_order
        self.number_states = number_states
        self.centroids = None
        self.coherence_matrix = None
        self.state_sequences = None

    def fit(self, time_coefficients, y):
        St_km, A_km, C = self.cluster_data(
            self.var_order, self.number_states, time_coefficients, y)
        self.centroids = np.array(C)
        self.coherence_matrix = np.array(A_km)
        self.state_sequence = np.array(St_km)

    def cluster_data(self, p, K, tvvar_vec, y):
        print(':: ACTIVITY: Applying a VNS model')
        #St_km, C, _, _ = self.engine.feval("kmindconnect.clustering.variable_neighbour_search.m", tvvar_vec, K, nout=4)
        St_km, C, _, _ = self.variable_neighbour_search(tvvar_vec, K)
        St_km, C = self.stabilize_states(St_km, C)
        N, T = y.shape
        # Pooling samples for regimes
        St = np.zeros((N, T, K))
        tj = np.zeros((K, 1))
        for j in np.arange(K):
            t = 0
            for i in np.arange(T):
                if St_km[i] == (j + 1):
                    St[:, t, j] = y[: , i]
                    t += 1
            tj[j] = t - 1

        print(':: ACTIVITY: Estimate state-specific VAR')
        # Estimate state - specific VAR
        A_km = np.zeros((N, N * p, K))
        for j in np.arange(K):
            A_km[:, :, j], _ = self.engine.feval("varfit.m", p, St[:, :int(tj[j]), j], nout=2)

        return St_km, A_km, C
        """
        function [St_km, A_km, C] = cluster_data(p, K, tvvar_vec, y)

            [St_km, C, ~, ~] = kmindconnect.clustering.variable_neighbour_search(tvvar_vec, K);

            [N, T] = size(y);
            % Pooling samples for regimes
            St = zeros(N, T, K);
            tj = zeros(K, 1);
            for j = 1:K
                t = 1;
                for i = 1:T
                    if St_km(i) == j
                        St(:, t, j) = y(:, i);
                        t = t + 1;
                    end
                end
                tj(j) = t - 1;
            end

            % Estimate state-specific VAR
            A_km = zeros(N, N * p, K);
            for j = 1:K
                [A_km(:, :, j), ~] = varfit(p, St(:, 1:tj(j), j));
            end
        end
        """

    def stabilize_states(self, St_km, C):
        sorted_indexes = np.argsort(np.sum(np.abs(C), axis=1))
        C = C[sorted_indexes]
        St_km0 = St_km.copy()
        for old, new_ in enumerate(sorted_indexes):
            St_km[np.where(St_km0 == (1 + new_))[0]] = old + 1
        return St_km, C

    def variable_neighbour_search(self, y, K):
        y = y.T
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
        C, St_km, D = kmeans(y, centres=C, metric="cityblock")
        return (1+St_km.reshape(-1, 1)), C, np.sum(D), np.min(D)
        """
        function [St_km, C, sumD, d] = variable_neighbour_search(tvvar_vec, K)
            [~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqeuclidean','Replicates',20);
            [~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqeuclidean','Start',C);
            [~, C, ~, d] = kmeans(tvvar_vec',K,'Distance','cityblock','Start',C);
            [~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqeuclidean','Start',C);
            [St_km, C, sumD, d] = kmeans(tvvar_vec',K,'Distance','cityblock','Start',C);
        end
        """

        def varfit(self, p, S):
            return var_model.VAR(S).fit(
                maxlags=p, method="ols", ic=None, trend="nc").params  # Fit a VAR
