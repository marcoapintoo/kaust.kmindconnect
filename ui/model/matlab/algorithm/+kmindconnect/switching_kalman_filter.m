function [A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, LL, St_skf, St_sks] = switching_kalman_filter(p, K, eps, ItrNo, A_km, y)

    [A0, H0, Q0, R0, x_0] = kmindconnect.kalman_filter.initialize_svar(p, K, A_km, y);
    [pi_, Z] = kmindconnect.kalman_filter.initialize_mtm(K);
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
        disp(sprintf(':: ACTIVITY: [%d] Initialize Kalman states', it))
        [x, P, S] = kmindconnect.kalman_filter.initialize_states(d_state, M, T, x_0, S, pi_);

        disp(sprintf(':: ACTIVITY: [%d] Applying Kalman filter', it))
        [xx_minus, PP_minus, Pe, e, L, S, xhat, Phat, xhat_full, Phat_full, Z] = kmindconnect.kalman_filter.kalman_filter(y, x, P, A, H, Q, R, L, xhat, Phat, xhat_full, Phat_full, xx_minus, PP_minus, S, Z);
        % Filtered state sequence
        fSt(2:T,:) = S(2:T,:);
        disp(sprintf(':: ACTIVITY: [%d] Applying Kalman smooter', it))
        [xs_t, xshat, Pshat, xshat_full, Pshat_full, U_t, S_MtT] = kmindconnect.kalman_filter.kalman_smoother(T, A, S, xhat, Phat, xhat_full, Phat_full, xshat, Pshat, PP_minus, xx_minus, Jt, Z);
        % Smoothed state sequence
        sSt = S_MtT(1:T,:);
        disp(sprintf(':: ACTIVITY: [%d] Calculate cross variance', it))
        P_ttm1T = kmindconnect.kalman_filter.get_cross_variance_terms(A, T, M, Phat, Jt, P_ttm1T);
        P_ttm1T_full = kmindconnect.kalman_filter.cross_collapse_cross_variance(T, M, d_state, U_t, xshat, xs_t, S_MtT, P_ttm1T);
        disp(sprintf(':: ACTIVITY: [%d] Calculate log-likelihood', it))
        LL(it) = kmindconnect.kalman_filter.log_likelihood(T, Pe, e, S, Z);
        DeltaL = abs((LL(it)-OldL)/LL(it)); % Stoping Criterion (Relative Improvement)
        %fprintf('  Improvement in L = %.2f\n',DeltaL);
        if(DeltaL < eps)
            break;
        end
        OldL = LL(it);
        disp(sprintf(':: ACTIVITY: [%d] Parameter optimization E-phase', it))
        [S_t, S_ttm1] = kmindconnect.kalman_filter.estimation_step(d_state, T, Pshat_full, P_ttm1T_full, xshat_full);
        disp(sprintf(':: ACTIVITY: [%d] Parameter optimization M-phase', it))
        [Q, R] = kmindconnect.kalman_filter.maximization_step(p, M, d_state, A, Q, H, R, y, S_MtT, S_ttm1, S_t, xshat_full);
    end
    Ahat=A; Qhat=Q; Rhat=R; Zhat=Z;

    Fxt = xhat_full;
    Sxt = xshat_full;
    
    disp(sprintf(':: ACTIVITY: [%d] Estimate state sequences', it))
    % Obtain estimated state sequence
    [~, St_skf] = max(fSt, [], 2);
    [~, St_sks] = max(sSt, [], 2);
end