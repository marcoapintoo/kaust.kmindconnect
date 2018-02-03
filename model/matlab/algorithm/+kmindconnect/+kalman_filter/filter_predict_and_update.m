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

