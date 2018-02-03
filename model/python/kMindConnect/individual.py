# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import linalg, fftpack

class IndividualProcessLogger:
    def __init__(self, associated_process):
        self.associated_process = associated_process

    def after_load_data(self):
        pass

    def after_get_time_coefficients(self):
        pass

    def after_get_cluster_coefficients(self):
        pass

    def after_get_state_space_coefficients(self):
        pass

    def before_load_data(self):
        pass

    def before_get_time_coefficients(self):
        pass

    def before_get_cluster_coefficients(self):
        pass

    def before_get_state_space_coefficients(self):
        pass


class IndividualProcess:
    def __init__(self, reader_engine, time_coeff_engine, clustering_engine, time_space_coeff_engine):
        self.logger = IndividualProcessLogger(self)
        # Process data
        self.data = None
        self.time_coefficients = None
        self.coherence_first_estimation = None
        self.centroids = None
        self.coherence_estimated = None
        self.partial_directed_coherence_estimated = None
        self.state_sequence = None
        self.state_sequence_filtered = None
        self.state_sequence_smoothed = None
        # Processing engines
        self.reader_engine = reader_engine
        self.time_coeff_engine = time_coeff_engine
        self.clustering_engine = clustering_engine
        self.time_space_coeff_engine = time_space_coeff_engine
    
    def load_data(self, path):
        self.logger.before_load_data()
        try:
            self.reader_engine.read(path)
            self.data = self.reader_engine.data
        finally:
            self.logger.after_load_data()
    
    def obtain_time_coefficients(self):
        self.logger.before_get_time_coefficients()
        try:
            self.time_coeff_engine.fit(self.data)
            self.time_coefficients = self.time_coeff_engine.coefficients
        finally:
            self.logger.after_get_time_coefficients()
    
    def cluster_coefficients(self):
        self.logger.before_get_cluster_coefficients()
        try:
            self.clustering_engine.fit(self.time_coefficients, self.data)
            self.centroids = self.clustering_engine.centroids
            self.coherence_first_estimation = self.clustering_engine.coherence_matrix
            self.state_sequence = self.clustering_engine.state_sequence
        finally:
            self.logger.after_get_cluster_coefficients()
    
    @classmethod
    def ar_to_pdc(cls, A):
        #https://gist.github.com/agramfort/9875439
        A = A.T
        p, N, N = A.shape
        n_fft = max(int(2 ** np.ceil(np.log2(p))), 512)
        A2 = np.zeros((n_fft, N, N))
        A2[1:p + 1, :, :] = A  # start at 1 !
        fA = fftpack.fft(A2, axis=0)
        freqs = fftpack.fftfreq(n_fft)
        I = np.eye(N)
        for i in range(n_fft):
            fA[i] = linalg.inv(I - fA[i])
        P = np.zeros((n_fft, N, N))
        sigma = np.ones(N)
        for i in range(n_fft):
            B = fA[i]
            B = linalg.inv(B)
            V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
            V = np.diag(V)  # denominator squared
            P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]
        return P


    def state_space_coefficients(self):
        self.logger.before_get_state_space_coefficients()
        try:
            self.time_space_coeff_engine.fit(self.coherence_first_estimation, self.data)
            self.coherence_estimated = self.time_space_coeff_engine.coherence_estimated
            self.partial_directed_coherence_estimated = self.ar_to_pdc(self.time_space_coeff_engine.coherence_estimated)
            self.state_sequence_filtered = self.time_space_coeff_engine.state_sequence_filtered
            self.state_sequence_smoothed = self.time_space_coeff_engine.state_sequence_smoothed
        finally:
            self.logger.after_get_state_space_coefficients()
    
    def run(self, path):
        self.load_data(path)
        self.obtain_time_coefficients()
        self.cluster_coefficients()
        self.state_space_coefficients()
        return self


"""
# OLD TESTING CLASSES
# Why was it deprecated? Short deadline and too fast changing API
#   
from .loaders import AutofileReader
from .matlab_engine import *
import matplotlib.pyplot as plt

class Testing:
    def __init__(self):
        self.reader_engine = AutofileReader(target="timeseries")
        self.time_coeff_engine = TVVAR()
        self.clustering_engine = Clustering()
        self.time_space_coeff_engine = SVAR()
        self.process = IndividualProcess(
            self.reader_engine,
            self.time_coeff_engine,
            self.clustering_engine,
            self.time_space_coeff_engine,
        )

    def test_loader(self, datafile):
        self.process.load_data(datafile)

    def test_time_coefficients(self, data_to_compare):
        self.process.obtain_time_coefficients()
        data = AutofileReader.get(data_to_compare)
        self.compare_matrices(self.process.time_coefficients,
                              data["time_varying_var_coefficients"])

    def test_clustering(self, data_to_compare):
        self.process.cluster_coefficients()
        data = AutofileReader.get(data_to_compare)
        self.process.clustering_engine.time_varying_states_var_coefficients = data[
            "time_varying_states_var_coefficients"]
        self.process.time_varying_states_var_coefficients = data[
            "time_varying_states_var_coefficients"]
        print(
            "1:", self.process.clustering_engine.time_varying_states_var_coefficients[:, :, 0])
        print(
            "2:", self.process.clustering_engine.time_varying_states_var_coefficients[:, :, 1])
        print(
            "3:", self.process.clustering_engine.time_varying_states_var_coefficients[:, :, 2])
        self.plot_clusters(self.process.clustering_engine.clustered_coefficients,
                           data["clustered_coefficients"].ravel() - 1)
        #self.compare_clusters(self.process.clustering_engine.clustered_coefficients, data["clustered_coefficients"].ravel() - 1)
        #self.compare_matrices(self.process.clustering_engine.cluster_centres, data["cluster_centres"])
        #self.compare_matrices(self.process.clustering_engine.expanded_time_series, data["expanded_time_series"])
        #self.compare_matrices(self.process.clustering_engine.length_by_cluster, data["length_by_cluster"])
        #self.compare_matrices(self.process.clustering_engine.time_varying_states_var_coefficients, data["time_varying_states_var_coefficients"])

    def test_state_space_coefficients(self, data_to_compare):
        self.process.state_space_coefficients()


from .loaders import AutofileReader
from .time_coeff import OriginalTVVAR
from .clustering import OriginalKMeans
from .time_space_coeff import OriginalSVAR
import matplotlib.pyplot as plt
class Testing2:
    def __init__(self):
        self.reader_engine = AutofileReader(target="timeseries")
        self.time_coeff_engine = OriginalTVVAR()
        self.clustering_engine = OriginalKMeans()
        self.time_space_coeff_engine = OriginalSVAR()
        self.process = IndividualProcess(
            self.reader_engine,
            self.time_coeff_engine,
            self.clustering_engine,
            self.time_space_coeff_engine,
        )
    
    def test_loader(self, datafile):
        self.process.load_data(datafile)
    
    
    def test_time_coefficients(self, data_to_compare):
        self.process.obtain_time_coefficients()
        data = AutofileReader.get(data_to_compare)
        self.compare_matrices(self.process.time_coefficients, data["time_varying_var_coefficients"])
    
    def test_clustering(self, data_to_compare):
        self.process.cluster_coefficients()
        data = AutofileReader.get(data_to_compare)
        self.process.clustering_engine.time_varying_states_var_coefficients = data["time_varying_states_var_coefficients"]
        self.process.time_varying_states_var_coefficients = data["time_varying_states_var_coefficients"]
        print("1:", self.process.clustering_engine.time_varying_states_var_coefficients[:, :, 0])
        print("2:", self.process.clustering_engine.time_varying_states_var_coefficients[:, :, 1])
        print("3:", self.process.clustering_engine.time_varying_states_var_coefficients[:, :, 2])
        self.plot_clusters(self.process.clustering_engine.clustered_coefficients, data["clustered_coefficients"].ravel() - 1)
        #self.compare_clusters(self.process.clustering_engine.clustered_coefficients, data["clustered_coefficients"].ravel() - 1)
        #self.compare_matrices(self.process.clustering_engine.cluster_centres, data["cluster_centres"])
        #self.compare_matrices(self.process.clustering_engine.expanded_time_series, data["expanded_time_series"])
        #self.compare_matrices(self.process.clustering_engine.length_by_cluster, data["length_by_cluster"])
        #self.compare_matrices(self.process.clustering_engine.time_varying_states_var_coefficients, data["time_varying_states_var_coefficients"])

    def test_state_space_coefficients(self, data_to_compare):
        self.process.state_space_coefficients()

    @staticmethod
    def plot_clusters(a, b):
        print("+" * 80, np.bincount(a), np.bincount(b))
        #
        a0, b0 = a.copy(), b.copy()
        for i, k in (enumerate(np.argsort(np.bincount(a)))):
            a[a0 == k] = i
        for i, k in enumerate(np.argsort(np.bincount(b))):
            b[b0 == k] = i
        #
        print("+" * 80, np.bincount(a), np.bincount(b))
        plt.figure(figsize=[10, 10], dpi=150)
        plt.subplot(3, 1, 1)
        plt.plot(a)
        plt.title("New implementation: clusters")
        plt.subplot(3, 1, 2)
        plt.plot(b)
        plt.title("Original: clusters")
        plt.subplot(3, 1, 3)
        plt.bar(np.arange(len(a)), a, 0.35,
                alpha=0.9,
                color='b',
                label='New')
        plt.bar(np.arange(len(b)) + 0.35, b, 0.35,
                alpha=0.9,
                color='r',
                label='Original')
        plt.title("Comparison: clusters")
        plt.savefig('cluster_comparison.png')
        plt.clf()
        plt.close()
        #

    @staticmethod
    def compare_clusters(a, b):
        if a.shape != b.shape:
            raise ValueError("Clusters have different shapes: {0} vs {1}!".format(a.shape, b.shape))
        a_normalized = np.sort(np.bincount(a))
        b_normalized = np.sort(np.bincount(b))
        differents = np.sum(a_normalized != b_normalized)
        if differents > 0:
            raise ValueError("There were recognized different clusters: {0} vs {1}!".format(a_normalized, b_normalized))

    @staticmethod
    def compare_matrices(a, b):
        if a.shape != b.shape:
            raise ValueError("Matrices have different shapes: {0} vs {1}!".format(a.shape, b.shape))
        error = np.sum(np.abs(a - b))
        if error >= 1e-6:
            print("===>", a)
            print("===>", b)
            raise ValueError("Matrices are too different! Found error: {0:.3f}".format(error))


if __name__ == '__main__': 
    test = Testing()
    #test.test_loader("/Users/pinto.marco/KAUSTProjects/TonitruumUI0/SVAR-simplified/SVAR-movie-fMRI-2/data/FS_16ROI_mean/6791_mean_fs.mat")
    test.test_loader("/Users/pinto.marco/KAUSTProjects/TonitruumUI0/SVAR-simplified/SVAR-movie-fMRI-2/Test_6251_mean_fs_00.mat")
    test.test_time_coefficients("/Users/pinto.marco/KAUSTProjects/TonitruumUI0/SVAR-simplified/SVAR-movie-fMRI-2/Test_6251_mean_fs_01.mat")
    test.test_clustering("/Users/pinto.marco/KAUSTProjects/TonitruumUI0/SVAR-simplified/SVAR-movie-fMRI-2/Test_6251_mean_fs_02.mat")
    test.test_state_space_coefficients("/Users/pinto.marco/KAUSTProjects/TonitruumUI0/SVAR-simplified/SVAR-movie-fMRI-2/Test_6251_mean_fs_03.mat")

"""

