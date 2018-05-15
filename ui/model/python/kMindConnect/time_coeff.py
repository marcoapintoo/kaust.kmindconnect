import os
import pandas as pd
import numpy as np
import statsmodels.tsa.vector_ar.var_model as var_model

class OriginalTVVAR:
    def __init__(self):
        self.coefficients = None
    
    def fit(self, y):
        p = 1 #VAR model order
        K = 3 #Number of states
        wlen = 50 #Window length
        shift = 1 #Window shift
        min_r_len = 5 #Minimum length for each regime
        N, T = y.shape
        tvvar_vec = np.zeros((p * N ** 2, T)) #TV-VAR coeffs
        win = np.ones(wlen) #form a window

        # initialize the indexes
        indx = 0
        t = 0
        Yw = np.zeros((N, wlen))

        # Short-Time VAR Analysis
        while indx + wlen <= T:
            for i in range(N):
                Yw[i, :] = y[i, indx: indx + wlen] * win.T
            At = var_model.VAR(Yw.T).fit(maxlags=p, method="ols", ic=None, trend="nc").params #Fit a VAR
            tvvar_vec[:, t] = At[:].ravel() #update time-varying matrix
            indx = indx + shift
            t = t + 1 #update the indexes
        self.coefficients = tvvar_vec