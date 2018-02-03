import numpy as np
import scipy.signal
import statsmodels.tsa.vector_ar.var_model as var_model

class TVVAR:
    def __init__(self, engine, var_order, window_length, window_shift):
        self.engine = engine
        self.var_order = var_order
        self.window_length = window_length
        self.window_shift = window_shift
        self.coefficients = None
    
    def fit(self, y):
        St = self.time_variable_var(
            self.var_order, self.window_length, self.window_shift, y)
        self.coefficients = np.array(St)

    def time_variable_var(self, p, wlen, shift, y):
        N, T = y.shape

        tvvar_vec = np.zeros((p * N ** 2, T)) # TV - VAR coeffs
        win = scipy.signal.boxcar(wlen)  # form a window

        # initialize the indexes
        indx = 0
        t = 0
        Yw = np.zeros((N, wlen))

        # Short - Time VAR Analysis
        while indx + wlen <= T:
            # windowing
            for i in np.arange(N):
                Yw[i, :] = y[i, indx: indx + wlen] * win.T
            At, _ = self.engine.feval("varfit.m", p, Yw, nout=2)  # Fit a VAR
            tvvar_vec[:, t] = At[:].ravel() #update time-varying matrix
            indx = indx + shift
            t = t + 1 # update the indexes
        
        return tvvar_vec
        """
        function [tvvar_vec] = time_variable_var(p, wlen, shift, y)
            [N, T] = size(y);

            tvvar_vec = zeros(p * N ^ 2, T); %TV-VAR coeffs
            win = rectwin(wlen); % form a window

            % initialize the indexes
            indx = 0; t = 1;
            Yw = zeros(N, wlen);

            % Short-Time VAR Analysis
            while indx + wlen <= T
                % windowing
                for i = 1:N
                    Yw(i, :) = y(i, indx + 1:indx + wlen) .* win';
                end
                [At, ~] = varfit(p, Yw); % Fit a VAR
                tvvar_vec(:, t) = At(:); % update time-varying matrix
                indx = indx + shift;
                t = t + 1; % update the indexes
            end
        end
        """

        def varfit(self, p, S):
            return var_model.VAR(S).fit(
                maxlags=p, method="ols", ic=None, trend="nc").params  # Fit a VAR
