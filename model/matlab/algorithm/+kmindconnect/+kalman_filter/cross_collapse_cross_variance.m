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

