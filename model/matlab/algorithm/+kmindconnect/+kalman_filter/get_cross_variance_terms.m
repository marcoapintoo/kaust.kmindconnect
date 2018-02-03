function [P_ttm1T] = get_cross_variance_terms(A, T, M, Phat, Jt, P_ttm1T)
    % Cross-variance terms
    for t=(T-1):-1:2
        for k=1:M
            A_k = A(:,:,k);
            for j=1:M
                P_ttm1T(:,:,j,k,t) = Phat(:,:,j,t)*Jt(:,:,j,k,t-1)'+Jt(:,:,j,k,t)...
                    *(P_ttm1T(:,:,j,k,t+1)-A_k*Phat(:,:,j,t))*Jt(:,:,j,k,t-1)';  %V(t,t-1|T)_jk
            end
        end
    end
end

