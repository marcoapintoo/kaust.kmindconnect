import numpy as np
class TVVAR:
    def __init__(self, engine, var_order, window_length, window_shift):
        self.engine = engine
        self.var_order = var_order
        self.window_length = window_length
        self.window_shift = window_shift
        self.coefficients = None
    
    def fit(self, y):
        St = self.engine.feval("kmindconnect.time_variable_var.m",
            float(self.var_order), float(self.window_length), float(self.window_shift), y)
        self.coefficients = np.array(St)

