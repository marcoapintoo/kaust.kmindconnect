function [x, P, S] = initialize_states(d_state, M, T, x_0, S, pi_)
    x = zeros(d_state,M,T);
    P = zeros(d_state,d_state,M);

    % Initialize state parameter
    x(:,:,1) = repmat(x_0,1,M);
    P_0 = eye(d_state);
    for i=1:M
        P(:,:,i) = P_0;
    end
    S(1,:) = pi_;

end

