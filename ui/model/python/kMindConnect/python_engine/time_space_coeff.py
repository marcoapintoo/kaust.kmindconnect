import numpy as np
from scipy.linalg import pinv

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
        (A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, Ls, St_skf, St_sks) = self.switching_kalman_filter(
            self.var_order, self.number_states, self.em_tolerance, self.em_max_iterations, coherence_estimation, y)
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


    def switching_kalman_filter(self, p, K, eps, ItrNo, A_km, y):

        #[A0, H0, Q0, R0, x_0] = self.engine.feval("kmindconnect.kalman_filter.initialize_svar.m", p, K, A_km, y, nout=5)
        A0, H0, Q0, R0, x_0 = self.initialize_svar(p, K, A_km, y)
        #[pi, Z] = self.engine.feval("kmindconnect.kalman_filter.initialize_mtm.m", K, nout=2)
        pi, Z = self.initialize_mtm(K)

        y = y.T
        A = A0.copy()
        H = H0.copy()
        Q = Q0.copy()
        R = R0.copy()
        T = y.shape[0]
        d_state = A.shape[0]
        M = Z.shape[0]

        # EM iteration
        OldL = 0
        LL = np.zeros(ItrNo, dtype="f8")
        S = np.zeros((T, M), dtype="f8")
        fSt = np.zeros((T, M), dtype="f8")
        xhat = np.zeros((d_state, M, T), dtype="f8")
        Phat = np.zeros((d_state, d_state, M, T), dtype="f8")
        Phat_full = np.zeros((d_state, d_state, T), dtype="f8")
        xhat_full = np.zeros((d_state, T), dtype="f8")
        xx_minus = np.zeros((d_state, M, M, T), dtype="f8")
        PP_minus = np.zeros((d_state, d_state, M, M, T), dtype="f8")
        Jt = np.zeros((d_state, d_state, M, M, T), dtype="f8")
        L = np.zeros((M, M, T), dtype="f8")
        P_ttm1T = np.zeros((d_state, d_state, M, M, T), dtype="f8")

        xshat = np.zeros((d_state, M, T), dtype="f8")
        Pshat = np.zeros((d_state, d_state, M, T), dtype="f8")

        for it in np.arange(0, ItrNo):
            print(':: ACTIVITY: [%d] Initialize Kalman states'%it)
            #[x, P, S] = self.engine.feval("kmindconnect.kalman_filter.initialize_states.m", d_state, M, T, x_0, S, pi, nout=3)
            x, P, S = self.initialize_states(d_state, M, T, x_0, S, pi)
            print(':: ACTIVITY: [%d] Applying Kalman filter' % it)
            [xx_minus, PP_minus, Pe, e, L, S, xhat, Phat, xhat_full, Phat_full, Z] = self.engine.feval("kmindconnect.kalman_filter.kalman_filter.m", y, x, P, A, H, Q, R, L, xhat, Phat, xhat_full, Phat_full, xx_minus, PP_minus, S, Z, nout=11)
            #[xx_minus, PP_minus, Pe, e, L, S, xhat, Phat, xhat_full, Phat_full, Z] = self.kalman_filter(y, x, P, A, H, Q, R, L, xhat, Phat, xhat_full, Phat_full, xx_minus, PP_minus, S, Z)
            # Filtered state sequence
            fSt[1:T, :] = S[1:T, :]
            print(':: ACTIVITY: [%d] Applying Kalman smooter' % it)
            [xs_t, xshat, Pshat, xshat_full, Pshat_full, U_t, S_MtT] = self.engine.feval("kmindconnect.kalman_filter.kalman_smoother.m", T, A, S, xhat, Phat, xhat_full, Phat_full, xshat, Pshat, PP_minus, xx_minus, Jt, Z, nout=7)
            # Smoothed state sequence
            sSt = S_MtT[:T,:]
            print(':: ACTIVITY: [%d] Calculate cross variance' % it)
            #P_ttm1T = self.engine.feval("kmindconnect.kalman_filter.get_cross_variance_terms.m", A, T, M, Phat, Jt, P_ttm1T)
            P_ttm1T = self.get_cross_variance_terms(A, T, M, Phat, Jt, P_ttm1T)
            #P_ttm1T_full = self.engine.feval("kmindconnect.kalman_filter.cross_collapse_cross_variance.m", T, M, d_state, U_t, xshat, xs_t, S_MtT, P_ttm1T)
            P_ttm1T_full = self.cross_collapse_cross_variance(T, M, d_state, U_t, xshat, xs_t, S_MtT, P_ttm1T)
            print(':: ACTIVITY: [%d] Calculate log-likelihood' % it)
            #LL[it] = self.engine.feval("kmindconnect.kalman_filter.log_likelihood.m", T, Pe, e, S, Z)
            LL[it] = self.log_likelihood(T, Pe, e, S, Z)
            DeltaL = np.abs((LL[it]-OldL)/LL[it]) # Stoping Criterion (Relative Improvement)
            if DeltaL < eps:
                break
            OldL = LL[it]
            print(':: ACTIVITY: [%d] Parameter optimization E-phase' % it)
            #[S_t, S_ttm1] = self.engine.feval("kmindconnect.kalman_filter.estimation_step.m", d_state, T, Pshat_full, P_ttm1T_full, xshat_full, nout=2)
            S_t, S_ttm1 = self.estimation_step(d_state, T, Pshat_full, P_ttm1T_full, xshat_full)
            print(':: ACTIVITY: [%d] Parameter optimization M-phase'%it)
            #[Q, R] = self.engine.feval("kmindconnect.kalman_filter.maximization_step.m", p, M, d_state, A, Q, H, R, y, S_MtT, S_ttm1, S_t, xshat_full, nout=2)
            Q, R = self.maximization_step(p, M, d_state, A, Q, H, R, y, S_MtT, S_ttm1, S_t, xshat_full)

        Ahat = A
        Qhat = Q
        Rhat = R
        Zhat = Z

        Fxt = xhat_full
        Sxt = xshat_full

        # Obtain estimated state sequence
        St_skf = fSt.max(axis=1).reshape(-1, 1)
        St_sks = sSt.max(axis=1).reshape(-1, 1)
        #print("... %s %s" %(St_skf, St_sks))
        #print("####", A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, LL, St_skf, St_sks)
        return A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, LL, St_skf, St_sks

        """
        function [A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, LL, St_skf, St_sks] = switching_kalman_filter(p, K, eps, ItrNo, A_km, y)

            [A0, H0, Q0, R0, x_0] = kmindconnect.kalman_filter.initialize_svar(p, K, A_km, y);
            [pi, Z] = kmindconnect.kalman_filter.initialize_mtm(K);

            y = y';
            A=A0; H=H0; Q=Q0; R=R0;
            [T, ~] = size(y);
            d_state = size(A,1);
            M = size(Z,1);

            % EM iteration
            OldL = 0;
            LL = zeros(ItrNo);
            S = zeros(T,M); 
            xhat = zeros(d_state,M,T);
            Phat = zeros(d_state,d_state,M,T);
            Phat_full  = zeros(d_state,d_state,T);
            xhat_full  = zeros(d_state,T);
            xx_minus = zeros(d_state,M,M,T);
            PP_minus = zeros(d_state,d_state,M,M,T);
            Jt = zeros(d_state,d_state,M,M,T);
            L = zeros(M,M,T);
            P_ttm1T = zeros(d_state,d_state,M,M,T);

            xshat = zeros(d_state,M,T);
            Pshat = zeros(d_state,d_state,M,T);


            for it=1:1:ItrNo
                [x, P, S] = kmindconnect.kalman_filter.initialize_states(d_state, M, T, x_0, S, pi);

                [xx_minus, PP_minus, Pe, e, L, S, xhat, Phat, xhat_full, Phat_full, Z] = kmindconnect.kalman_filter.kalman_filter(y, x, P, A, H, Q, R, L, xhat, Phat, xhat_full, Phat_full, xx_minus, PP_minus, S, Z);
                % Filtered state sequence
                fSt(2:T,:) = S(2:T,:);
                [xs_t, xshat, Pshat, xshat_full, Pshat_full, U_t, S_MtT] = kmindconnect.kalman_filter.kalman_smoother(T, A, S, xhat, Phat, xhat_full, Phat_full, xshat, Pshat, PP_minus, xx_minus, Jt, Z);
                % Smoothed state sequence
                sSt = S_MtT(1:T,:);
                P_ttm1T = kmindconnect.kalman_filter.get_cross_variance_terms(A, T, M, Phat, Jt, P_ttm1T);
                P_ttm1T_full = kmindconnect.kalman_filter.cross_collapse_cross_variance(T, M, d_state, U_t, xshat, xs_t, S_MtT, P_ttm1T);
                LL(it) = kmindconnect.kalman_filter.log_likelihood(T, Pe, e, S, Z);
                DeltaL = abs((LL(it)-OldL)/LL(it)); % Stoping Criterion (Relative Improvement)
                fprintf('  Improvement in L = %.2f\n',DeltaL);
                if(DeltaL < eps)
                    break;
                end
                OldL = LL(it);
                [S_t, S_ttm1] = kmindconnect.kalman_filter.estimation_step(d_state, T, Pshat_full, P_ttm1T_full, xshat_full);
                [Q, R] = kmindconnect.kalman_filter.maximization_step(p, M, d_state, A, Q, H, R, y, S_MtT, S_ttm1, S_t, xshat_full);
            end
            Ahat=A; Qhat=Q; Rhat=R; Zhat=Z;

            Fxt = xhat_full;
            Sxt = xshat_full;

            % Obtain estimated state sequence
            [~, St_skf] = max(fSt, [], 2);
            [~, St_sks] = max(sSt, [], 2);
        end
        """

    def initialize_svar(self, p, K, A_km, y):
        N, _ = y.shape

        A0 = np.zeros((N * p, N * p, K))
        H0 = np.zeros((N, N * p, K))
        Q0 = np.zeros((N * p, N * p, K))
        R0 = np.zeros((N, N, K))
        x_0 = np.zeros((N * p, 1))

        # Initialize SVAR parameters
        for j in np.arange(K):
            A0[:N, :, j] = A_km[:, :, j]
            for k in np.arange(p):
                if k < p - 1:
                    #A0[k * N: k * N + N, (k - 1) * N : (k - 1) * N + N, j] = np.eye(N)
                    A0[(k+1) * N: (k+1) * N + N, k * N : k * N + N, j] = np.eye(N)
            Q0[:N, :N, j] = np.eye(N)
            H0[:N, :N, j] = np.eye(N)
            R0[:, :, j] = 0.1 * np.eye(N)
        
        return A0, H0, Q0, R0, x_0
        """
        function [A0, H0, Q0, R0, x_0] = initialize_svar(p, K, A_km, y)
            [N, ~] = size(y);

            A0 = zeros(N * p, N * p, K);
            H0 = zeros(N, N * p, K);
            Q0 = zeros(N * p, N * p, K);
            R0 = zeros(N, N, K);
            x_0 = zeros(N * p, 1);

            % Initialize SVAR parameters
            for j = 1:K
                A0(1:N, :, j) = A_km(:, :, j);
                for k = 1:p
                    if k < p
                        A0(k * N + 1:k * N + N, (k - 1) * N + 1:(k - 1) * N + N, j) = eye(N, N);
                    end
                end
                Q0(1:N, 1:N, j) = eye(N);
                H0(1:N, 1:N, j) = eye(N);
                R0(:, :, j) = 0.1 * eye(N);
            end

        end
        """

    def initialize_mtm(self, K):
        # Markov Transition matrix
        pi_ = np.ones((1, K)) / K

        Z = 0.05 / (K - 1) * np.ones((K, K))
        Z.ravel()[np.r_[0: (K**2) : (K+1)]] = 0.95
        return pi_, Z
        """
        function [pi_, Z] = initialize_mtm(K)
            % Markov Transition matrix
            pi_ = ones(1, K) / K;

            Z = 0.05 / (K - 1) * ones(K, K);
            Z(1:K + 1:end) = 0.95;
        end
        """

    def initialize_states(self, d_state, M, T, x_0, S, pi_):
        x = np.zeros((d_state, M, T))
        P = np.zeros((d_state, d_state, M))

        # Initialize state parameter
        x[: , : , 0] = self.engine.feval("repmat", x_0, 1, M)
        P_0 = np.eye(d_state)
        for i in np.arange(M):
            P[:, : , i] = P_0
        S[0, :] = pi_
        return x, P, S
        """
        function [x, P, S] = initialize_states(d_state, M, T, x_0, S, pi_)
            x = zeros(d_state,M,T);
            P = zeros(d_state,d_state,M);

            % Initialize state parameter
            x(:,:,1) = repmat(x_0,1,M);
            P_0 = eye(d_state);
            for i=1:M
                P(:,:,i) = P_0;
            end
            S(1,:) = pi_;

        end
        """

    def kalman_filter(self, y, x, P, A, H, Q, R, L, xhat, Phat, xhat_full, Phat_full, xx_minus, PP_minus, S, Z):
        T, d_obs = y.shape
        M = Z.shape[0]
        d_state = A.shape[0]
        Pe = np.zeros((d_obs, d_obs, M, M, T))
        e = np.zeros((d_obs, M, M, T))
        x_ij = np.zeros((d_state, M, M))
        P_ij = np.zeros((d_state, d_state, M, M))
        P_ttm1T = np.zeros((d_state, d_state, M, M, T))
        for t in np.arange(1, T):
            S_norm = 0
            [xx_minus, PP_minus, Pe, e, x_ij, P_ij, L, S_marginal, S_norm] = self.engine.feval("kmindconnect.kalman_filter.filter_predict_and_update.m", d_state, M, T, t+1, y, x, Z, S, P, A, H, Q, R, L, Phat, xx_minus, PP_minus, Pe, e, x_ij, P_ij, P_ttm1T, S_norm, nout=9)
            #xx_minus, PP_minus, Pe, e, x_ij, P_ij, L, S_marginal, S_norm = self.filter_predict_and_update(d_state, M, T, t, y, x, Z, S, P, A, H, Q, R, L, Phat, xx_minus, PP_minus, Pe, e, x_ij, P_ij, P_ttm1T, S_norm)

            # Filtered occupancy probability of state j at time t
            S_marginal = S_marginal / S_norm
            
            # P(S(t)=j, S(t - 1)=i | y(1: t))
            for j in np.arange(M):
                S[t, j] = np.sum(S_marginal[: , j])
                # P(S(t)=j | y(1: t))(Eq. 16)
            
            # Weights of state components
            W = np.zeros((M, M))
            for j in np.arange(M):
                for i in np.arange(M):
                    W[i, j] = S_marginal[i, j] / S[t, j]
                    # P(S(t - 1)=i | S(t)=j, y(1: t))
            
            # Collapsing: Gaussian approximation
            for j in np.arange(M):
                x[: , j, t] = np.dot(x_ij[: , : , j], W[: , j].reshape(-1, 1)).ravel()
                P[:, : , j] = np.zeros((d_state, d_state))
                for i in np.arange(M):
                    m = x_ij[:, i, j] - x[: , j, t]
                    P[: , : , j] = P[: , : , j] + np.dot(W[i, j], P_ij[: , : , i, j] + np.dot(m, m.T))
                # Filtered density of x(t) given state j
                xhat[: , j, t] = x[: , j, t]
                # E(x(t) | S(t)=j, y(1: t))(Eq. 11)
                Phat[:, : , j, t] = P[: , : , j]
                # Cov(x(t) | S(t)=j, y(1: t))(Eq. 12)
            
            # Filtered density of x(t)
            for j in np.arange(M):
                xhat_full[:, t] = xhat_full[: , t] + np.dot(xhat[: , j, t], S[t, j])
                # E(x(t) | y(1: t))
            for j in np.arange(M):
                mu = xhat[: , j, t] - xhat_full[: , t]
                Phat_full[: , : , t] = Phat_full[: , : , t] + np.dot(S[t, j], Phat[: , : , j, t] + np.dot(mu, mu.T))
                # Cov(x(t) | y(1: t))
        #% End for t = 2: T
        return xx_minus, PP_minus, Pe, e, L, S, xhat, Phat, xhat_full, Phat_full, Z
        """
        function [xx_minus, PP_minus, Pe, e, L, S, xhat, Phat, xhat_full, Phat_full, Z] = kalman_filter(y, x, P, A, H, Q, R, L, xhat, Phat, xhat_full, Phat_full, xx_minus, PP_minus, S, Z)
            [T, d_obs] = size(y);
            M = size(Z,1);
            d_state = size(A,1);
            Pe = zeros(d_obs,d_obs,M,M,T);
            e  = zeros(d_obs,M,M,T);
            x_ij = zeros(d_state,M,M);
            P_ij = zeros(d_state,d_state,M,M);
            P_ttm1T = zeros(d_state,d_state,M,M,T);
            for t=2:T
                S_norm = 0;

                [xx_minus, PP_minus, Pe, e, x_ij, P_ij, L, S_marginal, S_norm] = kmindconnect.kalman_filter.filter_predict_and_update(d_state, M, T, t, y, x, Z, S, P, A, H, Q, R, L, Phat, xx_minus, PP_minus, Pe, e, x_ij, P_ij, P_ttm1T, S_norm);
                
                % Filtered occupancy probability of state j at time t
                S_marginal = S_marginal/S_norm; % P(S(t)=j,S(t-1)=i|y(1:t))
                for j=1:M
                    S(t,j) = sum(S_marginal(:,j)); % P(S(t)=j|y(1:t))      (Eq. 16)
                end

                % Weights of state components
                W = zeros(M,M);
                for j=1:M
                    for i=1:M
                        W(i,j) = S_marginal(i,j)/S(t,j); % P(S(t-1)=i|S(t)=j,y(1:t))
                    end
                end
                
                % Collapsing: Gaussian approximation
                for j=1:M
                    x(:,j,t) = x_ij(:,:,j) * W(:,j);
                    P(:,:,j) = zeros(d_state,d_state);
                    for i=1:M
                        m = x_ij(:,i,j) - x(:,j,t);
                        P(:,:,j) = P(:,:,j) + W(i,j)*(P_ij(:,:,i,j) + m*m');
                        clear m;
                    end
                    % Filtered density of x(t) given state j
                    xhat(:,j,t) = x(:,j,t);   % E(x(t)|S(t)=j,y(1:t))     (Eq. 11)
                    Phat(:,:,j,t) = P(:,:,j); % Cov(x(t)|S(t)=j,y(1:t))   (Eq. 12)
                end
                
                % Filtered density of x(t)
                for j=1:M
                    xhat_full(:,t) = xhat_full(:,t) + xhat(:,j,t) * S(t,j); % E(x(t)|y(1:t))
                end
                %disp('================= F')
                for j=1:M
                    mu = xhat(:,j,t) - xhat_full(:,t);
                    Phat_full(:,:,t) = Phat_full(:,:,t) + S(t,j)*(Phat(:,:,j,t) + mu*mu');  % Cov(x(t)|y(1:t))
                end
                clear S_marginal W;
            end % End for t=2:T

        end
        """

    def kalman_smoother(self, T, A, S, xhat, Phat, xhat_full, Phat_full, xshat, Pshat,  PP_minus, xx_minus, Jt, Z):
        return xs_t, xshat, Pshat, xshat_full, Pshat_full, U_t, S_MtT
        """
        function [xs_t, xshat, Pshat, xshat_full, Pshat_full, U_t, S_MtT] = kalman_smoother(T, A, S, xhat, Phat, xhat_full, Phat_full, xshat, Pshat,  PP_minus, xx_minus, Jt, Z)
            M = size(Z,1);
            d_state = size(A,1);
            xs_t = zeros(d_state,M,M,T);
            Ps_t = zeros(d_state,d_state,M,M,T);
            S_MtT(T,:) = S(T,:);
            xshat(:,:,T)   = xhat(:,:,T);
            Pshat(:,:,:,T) = Phat(:,:,:,T);

            %Pshat_full = zeros(d_state,d_state,T);
            %xshat_full = zeros(d_state,T);

            xshat_full(:,T)   = xhat_full(:,T);
            Pshat_full(:,:,T) = Phat_full(:,:,T);
            S_Mttp1T = zeros(M,M,T);
            xs = zeros(d_state,M,M);
            Ps = zeros(d_state,d_state,M,M);
            U_t = zeros(M,M,T);
            for t=T-1:-1:1
                S_n = zeros(M,1);
                S_m = zeros(M,M);
                
                for k=1:M
                    A_k = A(:,:,k);
                    for j=1:M
                        Jt(:,:,j,k,t) = Phat(:,:,j,t) * A_k' * pinv(PP_minus(:,:,j,k,t+1), 1e-07); %J(t)
                        xs(:,j,k) = xhat(:,j,t) + Jt(:,:,j,k,t)*(xshat(:,k,t+1) - A_k*xx_minus(:,j,k,t+1)); %X(t|T)
                        Ps(:,:,j,k) = Phat(:,:,j,t) + Jt(:,:,j,k,t)*(Pshat(:,:,k,t+1) - PP_minus(:,:,j,k,t+1)) * Jt(:,:,j,k,t)';    %V(t|T)
                        xs_t(:,j,k,t) = xs(:,j,k);
                        Ps_t(:,:,j,k,t) = Ps(:,:,j,k);
                        S_m(j,k) = S(t,j) * Z(j,k);
                    end
                end
                
                for k=1:M
                    for j=1:M
                        S_n(k,1) = S_n(k,1) + S_m(j,k);
                    end
                end
                
                for k=1:M
                    for j=1:M
                        U(j,k) = S_m(j,k)/S_n(k,1);
                        U_t(j,k,t) = U(j,k);
                    end
                end
                
                for k=1:M
                    for j=1:M
                        S_Mttp1T(j,k,t+1) = U(j,k)*S_MtT(t+1,k);
                    end
                end
                
                % Smoothed occupancy probability of state j at time t
                for j=1:M
                    S_MtT(t,j) = sum(S_Mttp1T(j,:,t+1));
                end
                for j=1:M
                    for k=1:M
                        W_2(k,j)= S_Mttp1T(j,k,t+1)/S_MtT(t,j); % P(S(t+1)=k|S(t)=j,y(1:T))
                    end
                end
                
                % Collapsing
                xshat_j = zeros(d_state,M);
                Pshat_j = zeros(d_state,d_state,M);
                for j=1:M
                    for k=1:M
                        xshat_j(:,j) = xshat_j(:,j) + xs(:,j,k) * W_2(k,j);
                    end
                    for k=1:M
                        m2 = xs(:,j,k) - xshat_j(:,j);
                        Pshat_j(:,:,j) = Pshat_j(:,:,j) + W_2(k,j)*(Ps(:,:,j,k) + m2*m2');
                        clear m2;
                    end
                    % Smoothed density of x(t) given state j
                    xshat(:,j,t)   = xshat_j(:,j);     % E(x(t)|S(t)=j,y(1:T))    (Eq. 13)
                    Pshat(:,:,j,t) = Pshat_j(:,:,j);   % Cov(x(t)|S(t)=j,y(1:T))  (Eq. 14)
                end
                
                % Smoothed density of x(t)
                for j=1:M
                    xshat_full(:,t) = xshat_full(:,t) + xshat_j(:,j) * S_MtT(t,j); % E(x(t)|y(1:T))
                end
                for j=1:M
                    m3 = xshat_j(:,j) - xshat_full(:,t);
                    Pshat_full(:,:,t) = Pshat_full(:,:,t) + S_MtT(t,j)*(Pshat_j(:,:,j) + m3*m3'); % Cov(x(t)|y(1:T))
                    clear m3;
                end
            end

        end
        """

    def cross_collapse_cross_variance(self, T, M, d_state, U_t, xshat, xs_t, S_MtT, P_ttm1T):
        mu_y_k = np.zeros((d_state, M, T))
        P_ttm1T_full = np.zeros((d_state,d_state,T))
        P_ttm1T_k = np.zeros((d_state,d_state, M, T))
        # Cross-collapsing cross-variance
        for t in np.arange(T-1, 0, -1):
            for k in np.arange(M):
                mu_x = 0
                mu_y = 0
                # P_ttm1T_k(:,:,k,t) = np.zeros((d_state,d_state))
                for j in np.arange(M):
                    mu_x = mu_x + xshat[:,k,t] * U_t[j,k,t-1]
                    mu_y = mu_y + xs_t[:,j,k,t-1] * U_t[j,k,t-1]
                mu_y_k[:,k,t] = mu_y
                for j in np.arange(M):
                    P_ttm1T_k[:,:,k,t] = P_ttm1T_k[:,:,k,t] + U_t[j,k,t-1] * (P_ttm1T[:,:,j,k,t] + (xshat[:,k,t] - mu_x) * (xs_t[:,j,k,t-1] - mu_y).T) #(Eq. 15)
            mu_x = 0
            mu_y = 0
            for k in np.arange(M):
                mu_x = mu_x + xshat[:,k,t] * S_MtT[t,k]
                mu_y = mu_y + mu_y_k[:,k,t-1] * S_MtT[t,k]
            for k in np.arange(M):
                P_ttm1T_full[:,:,t] = P_ttm1T_full[:,:,t] + S_MtT[t,k] * (P_ttm1T_k[:,:,k,t] + (xshat[:,k,t] - mu_x) * (mu_y_k[:,k,t-1] - mu_y).T)
        return P_ttm1T_full
        """
        function [P_ttm1T_full] = cross_collapse_cross_variance(T, M, d_state, U_t, xshat, xs_t, S_MtT, P_ttm1T)
            mu_y_k = zeros(d_state, M, T);
            P_ttm1T_full = zeros(d_state,d_state,T);
            P_ttm1T_k = zeros(d_state,d_state, M, T);
            % Cross-collapsing cross-variance
            for t=T:-1:2
                for k=1:M
                    mu_x = 0;
                    mu_y = 0;
                    %P_ttm1T_k(:,:,k,t) = zeros(d_state,d_state);
                    for j=1:M
                        mu_x = mu_x + xshat(:,k,t)*U_t(j,k,t-1);
                        mu_y = mu_y + xs_t(:,j,k,t-1)*U_t(j,k,t-1);
                    end
                    mu_y_k(:,k,t) = mu_y;
                    for j=1:M
                        P_ttm1T_k(:,:,k,t) = P_ttm1T_k(:,:,k,t) + U_t(j,k,t-1)*(P_ttm1T(:,:,j,k,t) + (xshat(:,k,t)-mu_x)*(xs_t(:,j,k,t-1)-mu_y)'); %(Eq. 15)
                    end
                    clear mu_x mu_y;
                end
                mu_x = 0;
                mu_y = 0;
                for k=1:M
                    mu_x = mu_x + xshat(:,k,t) * S_MtT(t,k);
                    mu_y = mu_y + mu_y_k(:,k,t-1)  * S_MtT(t,k);
                end
                for k=1:M
                    P_ttm1T_full(:,:,t) = P_ttm1T_full(:,:,t) + S_MtT(t,k)*(P_ttm1T_k(:,:,k,t) + (xshat(:,k,t)-mu_x)*(mu_y_k(:,k,t-1)-mu_y)');
                end
                clear mu_x mu_y;
            end
        end


        """
    def get_cross_variance_terms(self, A, T, M, Phat, Jt, P_ttm1T):
        # Cross-variance terms
        for t in np.arange(T - 2, 0, -1):
            for k in np.arange(M):
                A_k = A[:,:,k]
                for j in np.arange(M):
                    P_ttm1T[:,:,j,k,t] = np.dot(Phat[:,:,j,t], Jt[:,:,j,k,t-1].T) + np.dot(
                        np.dot(
                            Jt[:,:,j,k,t],
                            (P_ttm1T[:,:,j,k,t+1] - np.dot(A_k, Phat[:,:,j,t]))
                        ), Jt[:,:,j,k,t-1].T) #V(t,t-1|T)_jk
        return P_ttm1T
        """
        function [P_ttm1T] = get_cross_variance_terms(A, T, M, Phat, Jt, P_ttm1T)
            % Cross-variance terms
            for t=(T-1):-1:2
                for k=1:M
                    A_k = A(:,:,k);
                    for j=1:M
                        P_ttm1T(:,:,j,k,t) = Phat(:,:,j,t)*Jt(:,:,j,k,t-1)'+...
                        Jt(:,:,j,k,t)*(...
                            P_ttm1T(:,:,j,k,t+1)-A_k*Phat(:,:,j,t)...
                        ) *Jt(:,:,j,k,t-1)';  %V(t,t-1|T)_jk
                    end
                end
            end
        end
        """

    def log_likelihood(self, T, Pe, e,S, Z):
        M = Z.shape[1]
        # P(y(t)|y(1:t-1) = sum_i sum_j [P(y(t),S(t)=j,S(t-1)=i|y(1:t-1))]
        # where P(y(t),S(t)=j,S(t-1)=i|y(1:t-1) =
        # P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)*P(S(t)=j|S(t-1)=i,y(1:t-1))*P(S(t-1)=i|y(1:t-1))
        # = L(i,j,t)*Z(i,j)*S(t-1,j) = S_marginal(i,j)
        Lt = 0
        for t in np.arange(1, T):
            Acc = 0
            for j in np.arange(0, M):
                for i in np.arange(0, M):
                    log_S_marg_ij = -0.5*(np.log(np.linalg.det(Pe[:,:,i,j,t])) - np.dot(np.dot(0.5*e[:,i,j,t].T, pinv(Pe[:,:,i,j,t], 1e-07)), e[:,i,j,t])) + np.log(Z[i,j]) + np.log(S[t-1,i])
                    Acc = Acc + np.exp(log_S_marg_ij)
            Lt = Lt + np.log(Acc)
        return Lt
        """
        function [Lt] = log_likelihood(T, Pe, e,S, Z)
            M = size(Z,1);
            % P(y(t)|y(1:t-1) = sum_i sum_j [P(y(t),S(t)=j,S(t-1)=i|y(1:t-1))]
            % where P(y(t),S(t)=j,S(t-1)=i|y(1:t-1) =
            % P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)*P(S(t)=j|S(t-1)=i,y(1:t-1))*P(S(t-1)=i|y(1:t-1))
            % = L(i,j,t)*Z(i,j)*S(t-1,j) = S_marginal(i,j)
            Lt = 0;
            for t=2:T
                Acc = 0;
                for j=1:M
                    for i=1:M
                        log_S_marg_ij = - 0.5*(log(det(Pe(:,:,i,j,t))) - 0.5*e(:,i,j,t)'*pinv(Pe(:,:,i,j,t), 1e-07)*e(:,i,j,t)) + log(Z(i,j)) + log(S(t-1,i));
                        Acc = Acc + exp(log_S_marg_ij);
                    end
                end
                Lt = Lt + log(Acc);
            end

        end
        """

    def estimation_step(self, d_state, T, Pshat_full, P_ttm1T_full, xshat_full):
        S_t = np.zeros((d_state,d_state,T))
        S_ttm1 = np.zeros((d_state,d_state,T))
        for t in np.arange(1, T):
            S_t[:,:,t] = Pshat_full[:,:,t] + np.dot(xshat_full[:,t], xshat_full[:,t].T) # (Eq. 18)
            S_ttm1[:,:,t] = P_ttm1T_full[:,:,t] + np.dot(xshat_full[:,t], xshat_full[:,t-1].T) # (Eq. 19)
        return S_t, S_ttm1
        """
        function [S_t, S_ttm1] = estimation_step(d_state, T, Pshat_full, P_ttm1T_full, xshat_full)
            S_t    = zeros(d_state,d_state,T);
            S_ttm1 = zeros(d_state,d_state,T);
            for t=2:T
                S_t(:,:,t)    = Pshat_full(:,:,t)   + xshat_full(:,t)*xshat_full(:,t)';       % (Eq. 18)
                S_ttm1(:,:,t) = P_ttm1T_full(:,:,t) + xshat_full(:,t)*xshat_full(:,t-1)';     % (Eq. 19)
            end
        end
        """

    def maximization_step(self, p, M, d_state, A, Q, H, R, y, S_MtT, S_ttm1, S_t, xshat_full):
        T, d_obs = y.shape
        SumA1 = np.zeros((d_state,d_state,M))
        SumA2 = np.zeros((d_state,d_state,M))
        SumQ1 = np.zeros((d_state,d_state,M))
        SumQ2 = np.zeros((d_state,d_state,M))
        SumW = np.zeros((1,M))
        SumR  = np.zeros((d_obs,d_obs))
        SumR2 = np.zeros((d_obs,d_obs))

        Wj_t = S_MtT[:T,:]

        for i in np.arange(M):
            SumW[0,i] = np.sum(Wj_t[1:T,i])
            for t in np.arange(1, M):
                SumA1[:,:,i] = SumA1[:,:,i] + Wj_t[t,i]*S_ttm1[:,:,t]
                SumA2[:,:,i] = SumA2[:,:,i] + Wj_t[t,i]*S_t[:,:,t-1]
                SumQ1[:,:,i] = SumQ1[:,:,i] + Wj_t[t,i]*S_t[:,:,t]
                SumQ2[:,:,i] = SumQ2[:,:,i] + Wj_t[t,i]*S_ttm1[:,:,t].T
        A = A.copy()
        R = R.copy()
        Q = Q.copy()
        Qa = np.zeros((d_state, d_state))
        for i in np.arange(M):
            A[:,:,i] = np.dot(SumA1[:,:,i], pinv(SumA2[:,:,i], 1e-07)) # (Eq. 20)
            Qa[:,:] = (1/SumW[0,i]) * (SumQ1[:,:,i] - np.dot(A[:,:,i], SumQ2[:,:,i])) # (Eq. 21)
            
            # Constrait A for VAR(p) factors
            A[d_obs:d_state,:d_state, i] = 0
            for k in np.arange(p):
                if k < p - 1:
                    A[(k+1) * d_obs: (k+2) * d_obs,
                    k*d_obs:(k+1)*d_obs,
                    i] = np.eye(d_obs)
            Q[:d_obs,:d_obs,i] = Qa[:d_obs,:d_obs]
        # Obs noise covariance matrix
        for i in np.arange(M):
            SumR[:,:] = 0
            for t in np.arange(T):
                SumR2[:,:] = 0
                for j in np.arange(M):
                    SumR2 = SumR2 + Wj_t[t, j] * np.dot(H[:,:,j], np.dot(xshat_full[:,t], y[t,:]))
                SumR = SumR + np.dot(y[t,:].T, y[t,:]) - SumR2
            R[:,:,i] = (1/T) * SumR
            #R[:,:,i] = np.diag(np.diag(R[:,:,i]))
            R[:, :, i] = self.engine.feval("diag", self.engine.feval("diag", R[:, :, i]))
            R[:,:,i] = self.engine.feval("nearestSPD", R[:,:,i])
        return Q, R
        """
        function [Q, R] = maximization_step(p, M, d_state, A, Q, H, R, y, S_MtT, S_ttm1, S_t, xshat_full)
            [T, d_obs] = size(y);
            SumA1 = zeros(d_state,d_state,M);
            SumA2 = zeros(d_state,d_state,M);
            SumQ1 = zeros(d_state,d_state,M);
            SumQ2 = zeros(d_state,d_state,M);
            SumW = zeros(1,M);
            SumR  = zeros(d_obs,d_obs);
            SumR2 = zeros(d_obs,d_obs);

            Wj_t = S_MtT(1:T,:);

            for i=1:M
                SumW(1,i) = sum(Wj_t(2:T,i));
                for t=2:T
                    SumA1(:,:,i) = SumA1(:,:,i) + Wj_t(t,i)*S_ttm1(:,:,t);
                    SumA2(:,:,i) = SumA2(:,:,i) + Wj_t(t,i)*S_t(:,:,t-1);
                    SumQ1(:,:,i) = SumQ1(:,:,i) + Wj_t(t,i)*S_t(:,:,t);
                    SumQ2(:,:,i) = SumQ2(:,:,i) + Wj_t(t,i)*S_ttm1(:,:,t)';
                end
            end

            Qa = zeros(d_state,d_state);
            for i=1:M
                A(:,:,i)= SumA1(:,:,i)*pinv(SumA2(:,:,i), 1e-07);                       % (Eq. 20)
                Qa(:,:) = (1/SumW(1,i))*(SumQ1(:,:,i) - A(:,:,i)*SumQ2(:,:,i));  % (Eq. 21)
                
                % Constrait A for VAR(p) factors
                A(d_obs+1:d_state,1:d_state,i) = 0;
                for k=1:p
                    if k<p
                        A(k*d_obs+1:k*d_obs+d_obs,(k-1)*d_obs+1:(k-1)*d_obs+d_obs,i) = eye(d_obs,d_obs);
                    end
                end
                Q(1:d_obs,1:d_obs,i) = Qa(1:d_obs,1:d_obs);
            end

            % Obs noise covariance matrix
            for i=1:M
                SumR(:,:) = 0;
                for t=1:T
                    SumR2(:,:) = 0;
                    for j=1:M
                        SumR2 = SumR2 + Wj_t(t,j)*H(:,:,j)*xshat_full(:,t)*y(t,:);
                    end
                    SumR = SumR + y(t,:)'*y(t,:) - SumR2;
                end
                R(:,:,i) = (1/T)*SumR;
                R(:,:,i) = diag(diag(R(:,:,i)));
                R(:,:,i) = nearestSPD(R(:,:,i));
            end
        end


        """

    def filter_predict_and_update(self, d_state, M, T, t, y, x, Z, S, P, A, H, Q, R, L, Phat, xx_minus, PP_minus, Pe, e, x_ij, P_ij, P_ttm1T, S_norm):
        S_marginal = np.zeros((M, M))
        I = np.eye(d_state)
        S_norm = 0
        for j in np.arange(M):
            A_j = A[:,:,j]
            H_j = H[:,:,j]
            Q_j = Q[:,:,j]
            R_j = R[:,:,j]
            for i in np.arange(M):
                # One-step ahead Prediction
                x_minus = np.dot(A_j, x[:,i,t-1])
                P_minus = np.dot(np.dot(A_j, P[:,:,i]), A_j.T) + Q_j
                xx_minus[:,i,j,t] = x_minus
                PP_minus[:,:,i,j,t] = P_minus
                
                # Prediction error
                Pe[:,:,i,j,t] = np.dot(np.dot(H_j, P_minus), H_j.T) + R_j
                e[:,i,j,t] = y[t, :].T - np.dot(H_j, x_minus)
                
                # Kalman Gain
                K = np.dot(np.dot(P_minus, H_j.T), pinv(1.23456e-6 + Pe[:,:,i,j,t], 1e-05))
                
                # Filtering update
                x_ij[:, i, j] = x_minus + K * e[:,i,j,t]
                P_ij[: ,:, i, j] = np.dot(I - K*H_j, P_minus)
                
                if t == T - 1:
                    P_ttm1T[:, :, i, j, t] = np.dot(np.dot(I-K * H_j, A_j), Phat[:,:,j,t-1])
                
                # Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
                a = self.engine.feval("squeeze", e[:, i, j, t].T)
                covar_ = self.engine.feval("squeeze", Pe[:, :, i, j, t])
                L[i,j,t] = (np.linalg.det(covar_)) ** -.5 * np.exp(-.5 * np.sum(( np.dot(a, pinv(1.23456e-6 + covar_, 1e-07)) * a), axis=1))
                
                S_marginal[i,j] = L[i,j,t] * Z[i,j] * S[t, i]
                S_norm = S_norm + S_marginal[i,j]
                
        return xx_minus, PP_minus, Pe, e, x_ij, P_ij, L, S_marginal, S_norm
        """
        function [xx_minus, PP_minus, Pe, e, x_ij, P_ij, L, S_marginal, S_norm] = filter_predict_and_update(d_state, M, T, t, y, x, Z, S, P, A, H, Q, R, L, Phat, xx_minus, PP_minus, Pe, e, x_ij, P_ij, P_ttm1T, S_norm)
            S_marginal = zeros(M,M);
            I = eye(d_state);
            S_norm = 0;
            for j=1:M
                A_j = A(:,:,j);
                H_j = H(:,:,j);
                Q_j = Q(:,:,j);
                R_j = R(:,:,j);
                for i=1:M
                    % One-step ahead Prediction
                    x_minus = A_j * x(:,i,t-1);
                    P_minus = A_j * P(:,:,i) * A_j' + Q_j;
                    xx_minus(:,i,j,t)   = x_minus;
                    PP_minus(:,:,i,j,t) = P_minus;
                    
                    % Prediction error
                    Pe(:,:,i,j,t) = H_j * P_minus * H_j' + R_j;
                    e(:,i,j,t)    = y(t,:)' - H_j * x_minus;
                    
                    % Kalman Gain
                    K = P_minus * H_j' * pinv(1.23456e-6 + Pe(:,:,i,j,t), 1e-05);
                    
                    % Filtering update
                    x_ij(:,i,j)   = x_minus + K * e(:,i,j,t);
                    P_ij(:,:,i,j) = (I - K*H_j)*P_minus;
                    
                    if t==T
                        P_ttm1T(:,:,i,j,t) = (I-K*H_j)*A_j*Phat(:,:,j,t-1);
                    end
                    
                    % Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
                    a = squeeze(e(:,i,j,t)'); covar_ = squeeze(Pe(:,:,i,j,t));
                    L(i,j,t) = (det(covar_))^-.5 * exp(-.5 * sum(((a*pinv(1.23456e-6 + covar_, 1e-07)).*a), 2));
                    
                    S_marginal(i,j) = L(i,j,t) * Z(i,j) * S(t-1,i);
                    S_norm = S_norm + S_marginal(i,j);
                    
                    clear x_minus P_minus a covar_ K;
                    
                end
                clear A_j H_j Q_j R_j
            end

        end


        """
