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

