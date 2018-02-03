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

