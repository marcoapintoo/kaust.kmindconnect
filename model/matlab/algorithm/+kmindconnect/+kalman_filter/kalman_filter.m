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

