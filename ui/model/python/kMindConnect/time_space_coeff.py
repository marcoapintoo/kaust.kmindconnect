import numpy as np
import scipy.linalg

def nearestSPD(A):
    # test for a square matrix A
    print("before", A)
    r, c = A.shape
    if r != c:
        raise ValueError("Matrix must be squared")
    elif r == 1 and A <= 0:
        # A was scalar and non-positive, so just return eps
        Ahat = 1e-10#eps
        return Ahat

    # symmetrize A into B
    B = (A + A.T) / 2

    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    U, Sigma, V = np.linalg.svd(B)  # $
    H = np.matmul(V, np.matmul(Sigma, V.T))

    # get Ahat in the above formula
    Ahat = (B + H) / 2

    # ensure symmetry
    Ahat = (Ahat + Ahat.T) / 2

    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    p = 1
    k = 0
    while p != 0:
        try:
            R = scipy.linalg.cholesky(Ahat, lower=False)
            p = np.linalg.matrix_rank(np.matmul(R, R.T)) + 1
        except:
            p = 0
        k = k + 1
        print("p =", p)
        if p != 0:
            # Ahat failed the chol test. It must have been just a hair off,
            # due to floating point trash, so it is simplest now just to
            # tweak by adding a tiny multiple of an identity matrix.
            mineig = np.real(np.min(np.linalg.eig(Ahat)[0]))
            if mineig == 0: break
            print("mineig", mineig)
            Ahat = Ahat + (-mineig * k ** 2 + np.spacing(mineig))* np.eye(*A.shape)
    print("after", Ahat)
    return Ahat

class OriginalSVAR:
    def __init__(self):
        self.cluster_centres = None
        self.clustered_coefficients = None
        self.expanded_time_series = None
        self.length_by_cluster = None
        self.time_varying_states_var_coefficients = None

    def fit(self, time_space_coeff_engine, y):
        eps = 0.0001 # Tolerated error in Markov Matrices
        ItrNo = 15 # Iteration number in Markov Matrices
        A_km = time_space_coeff_engine
        #print(A_km)
        p = 1  # VAR model order
        K = 3  # Number of states
        N, T = y.shape
        A0 = np.zeros((N * p, N * p, K))
        H0 = np.zeros((N, N * p, K))
        Q0 = np.zeros((N * p, N * p, K))
        R0 = np.zeros((N, N, K))
        x_0 = np.zeros(N * p)

        # Initialize SVAR parameters
        for j in range(K):
            A0[:N, :, j] = A_km[: , : , j]
            for k in range(p):
                if k < p - 1:
                    print(k, N)
                    print(k * N,":", k * N + N,",", (k ) * N,":", (k ) * N + N,",", j)
                    A0[k * N : k * N + N, k * N : k * N + N, j] = np.eye(N)
            Q0[:N, :N, j] = np.eye(N)
            H0[:N, :N, j] = np.eye(N)
            R0[:, :, j] = 0.1 * np.eye(N)
        # Markov Transition matrix
        pi = np.ones((1, K)) / K
        Z = 0.05 / (K - 1) * np.ones((K, K))
        Z[np.arange(0, len(Z), K + 1)] = 0.95

        #[fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, Ls] = varp_skf_em4(y', A0, H0, Q0, R0, x_0, Z, pi, p, eps, ItrNo);
        
        A, H, Q, R = A0, H0, Q0, R0
        y = y.T
        T, d_obs = y.shape
        d_state = A.shape[0]
        M = Z.shape[0]
        I = np.eye(d_state)

        # Declaring Kalman filter variables
        x_ij = np.zeros((d_state,M,M))
        P_ij = np.zeros((d_state,d_state,M,M))
        xhat = np.zeros((d_state,M,T))
        Phat = np.zeros((d_state,d_state,M,T))
        xx_minus = np.zeros((d_state,M,M,T))
        PP_minus = np.zeros((d_state,d_state,M,M,T))
        Pe = np.zeros((d_obs,d_obs,M,M,T))
        e  = np.zeros((d_obs,M,M,T))
        L = np.zeros((M,M,T))
        S = np.zeros((T,M))
        fSt = np.zeros((T,M))
        S_MtT = np.zeros((T,M))
        sSt = np.zeros((T,M))

        # Declaring Kalman smoothing variables
        xshat = np.zeros((d_state,M,T))
        Pshat = np.zeros((d_state,d_state,M,T))
        Jt = np.zeros((d_state,d_state,M,M,T))
        xs = np.zeros((d_state,M,M))
        Ps = np.zeros((d_state,d_state,M,M))
        xs_t = np.zeros((d_state,M,M,T))
        Ps_t = np.zeros((d_state,d_state,M,M,T))
        P_ttm1T = np.zeros((d_state,d_state,M,M,T))

        # Declaring E-step variables
        S_t    = np.zeros((d_state,d_state,T))
        S_ttm1 = np.zeros((d_state,d_state,T))

        # EM iteration
        OldL = 0
        LL =  np.zeros(ItrNo)
        for it in range(ItrNo):
            x = np.zeros((d_state,M,T))
            P = np.zeros((d_state,d_state,M))
            # Initialize state parameter
            print(x.shape, "=",d_state, M, x_0.shape)
            print(x[:,:,0].shape, ":", np.tile(x_0.reshape(-1,1),(1,M)).shape)
            x[:, :, 0] = np.tile(x_0.reshape(-1, 1), (1, M))
            P_0 = np.eye((d_state))
            for i in range(M):
                P[:,:,i] = P_0
            S[0,:] = pi

            P_ttm1T_full = np.zeros((d_state,d_state,T))
            Pshat_full = np.zeros((d_state,d_state,T))
            xshat_full = np.zeros((d_state,T))
            Phat_full  = np.zeros((d_state,d_state,T))
            xhat_full  = np.zeros((d_state,T))

            #-------------------------------------------------------------------------#
            #                        Switching Kalman Filter                          #
            #-------------------------------------------------------------------------#
            for t in range(1, T):
                S_norm = 0
                S_marginal = np.zeros((M,M))

                for j in range(M):
                    A_j = A[:,:,j]
                    H_j = H[:,:,j]
                    Q_j = Q[:,:,j]
                    R_j = R[:,:,j]

                    for i in range(M):
                        # One-step ahead Prediction
                        x_minus = np.matmul(A_j, x[:,i,t-1])
                        P_minus = np.matmul(A_j, np.matmul(P[:, :, i], A_j.T)) + Q_j
                        xx_minus[:,i,j,t]   = x_minus
                        PP_minus[:,:,i,j,t] = P_minus

                        # Prediction error
                        #print(d_obs, Pe[:,:,i,j,t].shape, H_j.shape, P_minus.shape)
                        #print(d_obs, Pe[:,:,i,j,t].shape, np.matmul(H_j,np.matmul( P_minus, H_j.T)).shape)
                        Pe[:,:,i,j,t] = np.matmul(H_j,np.matmul( P_minus, H_j.T)) + R_j
                        e[:,i,j,t]    = y[t,:].T - np.matmul(H_j, x_minus)

                        # Kalman Gain
                        #print(Pe[:,:,i,j,t][:10, :])
                        K = np.matmul(P_minus,np.matmul( H_j.T, np.linalg.pinv(Pe[:,:,i,j,t], 1e-05)))

                        # Filtering update
                        #print(x_minus.shape, K.shape, e[:, i, j, t].shape)
                        x_ij[:,i,j]   = x_minus + np.matmul(K, e[:,i,j,t])
                        P_ij[:,:,i,j] = np.matmul((I - np.matmul(K, H_j)), P_minus)

                        if t == T:
                            P_ttm1T[:,:,i,j,t] = np.matmul((I-np.matmul(K,H_j)), np.matmul(A_j, Phat[:,:,j,t-1]))

                        # Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
                        a = np.squeeze(e[:,i,j,t].T)
                        covar = np.squeeze(Pe[:,:,i,j,t])
                        #print("*", np.matmul(a, np.linalg.pinv(covar, 1e-07)).shape, np.linalg.pinv(covar, 1e-07).shape, a.shape)
                        #print(covar, Pe[:,:,i,j,t])
                        
                        L[i,j,t] = (np.linalg.det(covar) ** -.5) * np.exp(-.5 * np.sum(np.matmul(a, np.linalg.pinv(covar, 1e-07)) * a) )

                        S_marginal[i,j] = L[i,j,t] * Z[i,j] * S[t-1,i]
                        if False: print("  ",
                                  np.linalg.det(covar),
                                  np.sum(np.matmul(a, np.linalg.pinv(covar, 1e-07)) * a),
                                  np.exp(-.5 * np.sum(np.matmul(a, np.linalg.pinv(covar, 1e-07)) * a) )
                                  )
                        if S_marginal[i,j] == 0:
                            print(covar.shape, Pe[:,:,i,j,t].shape)
                            print("////", t, j, T,  L[i,j,t], Z[i,j], S[t-1,i])
                        S_norm = S_norm + S_marginal[i,j]
                        #print(S_norm)

                        #clear x_minus P_minus a covar K
                    #clear A_j H_j Q_j R_j
                
                # Filtered occupancy probability of state j at time t
                if S_norm == 0 or np.isnan(S_norm):
                    print("=", S_marginal, S_norm, S_marginal / S_norm, it, t, j, T)
                    sys.exit(0)
                S_marginal = S_marginal/S_norm # P(S(t)=j,S(t-1)=i|y[1:t])
                for j in range(M):
                    S[t,j] = np.sum(S_marginal[:,j]) # P(S(t)=j|y[1:t])      (Eq. 16))

                # Weights of state components
                W = np.zeros((M,M))
                for j in range(M):
                    for i in range(M):
                        W[i,j] = S_marginal[i,j]/S[t,j] # P(S(t-1)=i|S(t)=j,y[1:t])

                # Collapsing: Gaussian approximation
                for j in range(M):
                    x[:,j,t] = np.matmul(x_ij[:,:,j], W[:,j])
                    P[:,:,j] = np.zeros((d_state,d_state))
                    for i in range(M):
                        m = x_ij[:,i,j] - x[:,j,t]
                        P[:,:,j] = P[:,:,j] + W[i,j]*(P_ij[:,:,i,j] + np.matmul(m, m.T))
                        #clear m
                    # Filtered density of x(t) given state j
                    xhat[:,j,t] = x[:,j,t]   # E(x(t)|S(t)=j,y[1:t])     (Eq. 11)
                    Phat[:,:,j,t] = P[:,:,j] # Cov(x(t)|S(t)=j,y[1:t])   (Eq. 12)

                # Filtered density of x(t)
                for j in range(M):
                    xhat_full[:,t] = xhat_full[:,t] + xhat[:,j,t] * S[t,j] # E(x(t)|y[1:t])
                for j in range(M):
                    mu = xhat[:,j,t] - xhat_full[:,t]
                    Phat_full[:,:,t] = Phat_full[:,:,t] + S[t,j] * (Phat[:,:,j,t] + np.matmul(mu, mu.T))  # Cov(x(t)|y[1:t])
                #clear S_marginal W

            # Filtered state sequence
            fSt[1:T,:] = S[1:T,:]

            #-------------------------------------------------------------------------#
            #                        Switching Kalman Smoother                        #
            #-------------------------------------------------------------------------#
            S_MtT[T-1,:] = S[T-1,:]
            xshat[:,:,T-1]   = xhat[:,:,T-1]
            Pshat[:,:,:,T-1] = Phat[:,:,:,T-1]
            xshat_full[:,T-1]   = xhat_full[:,T-1]
            Pshat_full[:,:,T-1] = Phat_full[:,:,T-1]
            S_Mttp1T = np.zeros((M,M,T))

            W_2 = np.zeros((M, M))
            U = np.zeros((M, M))
            U_t = np.zeros((M, M, T-1))
            for t in range(T-2, -1, -1):
                S_n = np.zeros((M,1))
                S_m = np.zeros((M,M))

                for k in range(M):
                    A_k = A[:,:,k]
                    for j in range(M):
                        #print(t, k, j)
                        Jt[:,:,j,k,t] = np.matmul(Phat[:,:,j,t], np.matmul(A_k.T, np.linalg.pinv(PP_minus[:,:,j,k,t+1], 1e-07))) #J(t)
                        xs[:,j,k] = xhat[:,j,t] + np.matmul(Jt[:,:,j,k,t], (xshat[:,k,t+1] - np.matmul(A_k, xx_minus[:,j,k,t+1]))) #X(t|T)
                        Ps[:,:,j,k] = Phat[:,:,j,t] + np.matmul(Jt[:,:,j,k,t], np.matmul(Pshat[:,:,k,t+1] - PP_minus[:,:,j,k,t+1], Jt[:,:,j,k,t].T))    #V(t|T)
                        xs_t[:,j,k,t] = xs[:,j,k]
                        Ps_t[:,:,j,k,t] = Ps[:,:,j,k]
                        S_m[j,k] = S[t,j] * Z[j,k]
                for k in range(M):
                    for j in range(M):
                        S_n[k,0] = S_n[k,0] + S_m[j,k]
                for k in range(M):
                    for j in range(M):
                        U[j,k] = S_m[j,k]/S_n[k,0]
                        U_t[j,k,t] = U[j,k]
                for k in range(M):
                    for j in range(M):
                        S_Mttp1T[j,k,t+1] = U[j,k]*S_MtT[t+1,k]
                # Smoothed occupancy probability of state j at time t
                for j in range(M):
                    S_MtT[t,j] = np.sum(S_Mttp1T[j,:,t+1])
                for j in range(M):
                    for k in range(M):
                        W_2[k,j]= S_Mttp1T[j,k,t+1]/S_MtT[t,j] # P(S(t+1)=k|S(t)=j,y[1:T])
                # Collapsing
                xshat_j = np.zeros((d_state,M))
                Pshat_j = np.zeros((d_state,d_state,M))
                for j in range(M):
                    for k in range(M):
                        xshat_j[:,j] = xshat_j[:,j] + xs[:,j,k] * W_2[k,j]
                    for k in range(M):
                        m2 = xs[:,j,k] - xshat_j[:,j]
                        Pshat_j[:,:,j] = Pshat_j[:,:,j] + W_2[k,j] * (Ps[:,:,j,k] + np.matmul(m2, m2.T))
                        #clear m2
                    # Smoothed density of x(t) given state j
                    xshat[:,j,t]   = xshat_j[:,j]     # E(x(t)|S(t)=j,y[1:T])    (Eq. 13)
                    Pshat[:,:,j,t] = Pshat_j[:,:,j]   # Cov(x(t)|S(t)=j,y[1:T])  (Eq. 14)
                # Smoothed density of x(t)
                for j in range(M):
                    xshat_full[:,t] = xshat_full[:,t] + xshat_j[:,j] * S_MtT[t,j] # E(x(t)|y[1:T])
                for j in range(M):
                    m3 = xshat_j[:,j] - xshat_full[:,t]
                    Pshat_full[:,:,t] = Pshat_full[:,:,t] + S_MtT[t,j]*(Pshat_j[:,:,j] + np.matmul(m3, m3.T)) # Cov(x(t)|y[1:T])

            # Smoothed state sequence
            sSt = S_MtT[1:T,:]

            # Cross-variance terms
            for t in range(T-2,1,-1):
                for k in range(M):
                    A_k = A[:,:,k]
                    for j in range(M):
                        P_ttm1T[:,:,j,k,t] = np.matmul(Phat[:,:,j,t],
                            Jt[:,:,j,k,t-1].T + np.matmul(Jt[:,:,j,k,t], np.matmul(P_ttm1T[:,:,j,k,t+1]-A_k*Phat[:,:,j,t], Jt[:,:,j,k,t-1].T))
                        )  #V(t,t-1|T)_jk
            P_ttm1T_k = np.zeros((d_state, d_state, M, T))
            mu_y_k = np.zeros((d_state, M, T))
            # Cross-collapsing cross-variance
            for t in range(T-1, 1, -1):
                for k in range(M):
                    mu_x = 0
                    mu_y = 0
                    P_ttm1T_k[:,:,k,t] = np.zeros((d_state,d_state))
                    for j in range(M):
                        mu_x = mu_x + xshat[:,k,t]*U_t[j,k,t-1]
                        mu_y = mu_y + xs_t[:,j,k,t-1]*U_t[j,k,t-1]
                    mu_y_k[:,k,t] = mu_y
                    for j in range(M):
                        P_ttm1T_k[:,:,k,t] = P_ttm1T_k[:,:,k,t] + U_t[j,k,t-1] * (P_ttm1T[:,:,j,k,t] + (xshat[:,k,t]-mu_x)*(xs_t[:,j,k,t-1]-mu_y).T) #(Eq. 15)
                    #clear mu_x mu_y
                mu_x = 0
                mu_y = 0
                for k in range(M):
                    mu_x = mu_x + xshat[:,k,t] * S_MtT[t,k]
                    mu_y = mu_y + mu_y_k[:,k,t-1]  * S_MtT[t,k]
                for k in range(M):
                    P_ttm1T_full[:,:,t] = P_ttm1T_full[:,:,t] + S_MtT[t,k]*(P_ttm1T_k[:,:,k,t] + (xshat[:,k,t]-mu_x)*(mu_y_k[:,k,t-1]-mu_y).T)
                #clear mu_x mu_y

            #-------------------------------------------------------------------------#
            #                        Log-likelihood computation                       #
            #-------------------------------------------------------------------------#
            Lt = 0
            for t in range(1, T):
                Acc = 0
                for j in range(M):
                    for i in range(M):
                        log_S_marg_ij = - 0.5*(
                                np.log(np.linalg.det(Pe[:,:,i,j,t]))
                                - 0.5* np.matmul(e[:,i,j,t].T, np.matmul(np.linalg.pinv(Pe[:,:,i,j,t], 1e-07), e[:,i,j,t])) 
                                + np.log(Z[i,j]) + np.log(S[t-1,i]))
                        Acc = Acc + np.exp((log_S_marg_ij))
                Lt = Lt + np.log((Acc))
            print(Lt, LL.shape)
            LL[it] = Lt
            DeltaL = (LL[it]-OldL)/LL[it]
            print("DeltaL:", DeltaL)
            DeltaL = np.abs(DeltaL) # Stoping Criterion (Relative Improvement)
            if DeltaL < eps:
                ConvergeL = LL(it)
                break
            OldL = LL[it]

            #-------------------------------------------------------------------------#
            #                               E-step                                    #
            #-------------------------------------------------------------------------#
            for t in range(1, T):
                S_t[:,:,t]    = Pshat_full[:,:,t]   + np.matmul(xshat_full[:,t], xshat_full[:,t].T)       # (Eq. 18)
                S_ttm1[:,:,t] = P_ttm1T_full[:,:,t] + np.matmul(xshat_full[:,t], xshat_full[:,t-1].T)     # (Eq. 19)

            #-------------------------------------------------------------------------#
            #                               M-step                                    #
            #-------------------------------------------------------------------------#
            SumA1 = np.zeros((d_state,d_state,M))
            SumA2 = np.zeros((d_state,d_state,M))
            SumQ1 = np.zeros((d_state,d_state,M))
            SumQ2 = np.zeros((d_state,d_state,M))
            SumW = np.zeros((1,M))
            SumR  = np.zeros((d_obs,d_obs))
            SumR2 = np.zeros((d_obs,d_obs))

            Wj_t = S_MtT[:T,:]

            for i in range(M):
                SumW[0,i] = np.sum(Wj_t[:T,i])
                for t in range(1, T):
                    SumA1[:,:,i] = SumA1[:,:,i] + Wj_t[t,i]*S_ttm1[:,:,t]
                    SumA2[:,:,i] = SumA2[:,:,i] + Wj_t[t,i]*S_t[:,:,t-1]
                    SumQ1[:,:,i] = SumQ1[:,:,i] + Wj_t[t,i]*S_t[:,:,t]
                    SumQ2[:,:,i] = SumQ2[:,:,i] + Wj_t[t,i]*S_ttm1[:,:,t].T
            Qa = np.zeros((d_state,d_state))
            for i in range(M):
                A[:,:,i]= np.matmul(SumA1[:,:,i], np.linalg.pinv((SumA2[:,:,i]), 1e-07))                       # (Eq. 20)
                Qa[:,:] = (1/SumW[0,i])*(SumQ1[:,:,i] - np.matmul(A[:,:,i], SumQ2[:,:,i]))  # (Eq. 21)
                # Constrait A for VAR(p) factors
                A[d_obs+1:d_state,1:d_state,i] = 0
                for k in range(p-1):#DOUBT!
                    if k < p:
                        A[(k+1)*d_obs:(k+2)*d_obs,k*d_obs:k*d_obs+d_obs,i] = np.eye(d_obs)
                Q[:d_obs,:d_obs,i] = Qa[:d_obs,:d_obs]
                # Markov transition matrix
            # Obs noise covariance matrix
            for i in range(M):
                SumR[:,:] = 0
                for t in range(T):
                    SumR2[:,:] = 0
                    for j in range(M):
                        SumR2 = SumR2 + Wj_t[t,j] * H[:,:,j] * np.matmul(xshat_full[:,t], y[t,:])
                    SumR = SumR + np.matmul(y[t,:].T, y[t,:]) - SumR2
                R[:,:,i] = (1/T)*SumR
                R[:,:,i] = np.diag(np.diag(R[:,:,i]))
                print(R.shape, R[:,:,i].shape, nearestSPD(R[:,:,i]).shape)
                R[:,:,i] = nearestSPD(R[:,:,i])

            Ahat=A
            Qhat=Q
            Rhat=R
            Zhat=Z
        
        # Obtain estimated state sequence
        aF, St_skf = np.amax(fSt, axis=1), np.argmax(fSt, axis=1)
        aS, St_sks = np.amax(sSt, axis=1), np.argmax(fst, axis=1)

