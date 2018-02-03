import numpy as np

class Clustering:
    def __init__(self, engine, var_order, number_states):
        self.engine = engine
        self.var_order = var_order
        self.number_states = number_states
        self.centroids = None
        self.coherence_matrix = None
        self.state_sequences = None

    def fit(self, time_coefficients, y):
        St_km, A_km, C = self.engine.feval("kmindconnect.cluster_data.m",
            float(self.var_order), float(self.number_states), time_coefficients, y, nout=3)
        self.centroids = np.array(C)
        self.coherence_matrix = np.array(A_km)
        self.state_sequence = np.array(St_km)
    
