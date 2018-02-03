import numpy as np

class SVAR:
    def __init__(self, engine, var_order, number_states, em_tolerance, em_max_iterations):
        self.engine = engine
        self.var_order = var_order
        self.number_states = number_states
        self.em_tolerance = em_tolerance
        self.em_max_iterations = em_max_iterations
        self.initial_coherence = None
        self.initial_state_variance_error = None
        self.initial_signal_variance_error = None
        self.coherence_estimated = None
        self.state_variance_error_estimated = None
        self.signal_variance_error_estimated = None
        self.matrix_sequence_filtered = None
        self.matrix_sequence_smoothed = None
        self.matrix_factor_filtered = None
        self.matrix_factor_smoothed = None
        self.state_sequence_filtered = None
        self.state_sequence_smoothed = None

    def fit(self, coherence_estimation, y):
        (A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, Ls, St_skf, St_sks) = self.engine.feval("kmindconnect.switching_kalman_filter.m",
            float(self.var_order), float(self.number_states), float(self.em_tolerance), float(self.em_max_iterations), coherence_estimation, y, nout=14)
        self.initial_coherence = np.array(A0)
        self.initial_state_variance_error = np.array(Q0)
        self.initial_signal_variance_error = np.array(R0)
        self.coherence_estimated = np.array(Ahat)
        self.state_variance_error_estimated = np.array(Qhat)
        self.signal_variance_error_estimated = np.array(Rhat)
        self.matrix_sequence_filtered = np.array(fSt)
        self.matrix_sequence_smoothed = np.array(sSt)
        self.matrix_factor_filtered = np.array(Fxt)
        self.matrix_factor_smoothed = np.array(Sxt)
        self.state_sequence_filtered = np.array(St_skf)
        self.state_sequence_smoothed = np.array(St_sks)
