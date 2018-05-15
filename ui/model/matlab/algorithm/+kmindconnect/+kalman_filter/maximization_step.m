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

