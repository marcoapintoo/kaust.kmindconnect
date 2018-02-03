function [S_t, S_ttm1] = estimation_step(d_state, T, Pshat_full, P_ttm1T_full, xshat_full)
    S_t    = zeros(d_state,d_state,T);
    S_ttm1 = zeros(d_state,d_state,T);
    for t=2:T
        S_t(:,:,t)    = Pshat_full(:,:,t)   + xshat_full(:,t)*xshat_full(:,t)';       % (Eq. 18)
        S_ttm1(:,:,t) = P_ttm1T_full(:,:,t) + xshat_full(:,t)*xshat_full(:,t-1)';     % (Eq. 19)
    end
end

