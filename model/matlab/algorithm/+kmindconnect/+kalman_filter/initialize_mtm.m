function [pi_, Z] = initialize_mtm(K)

    % Markov Transition matrix
    pi_ = ones(1, K) / K;

    Z = 0.05 / (K - 1) * ones(K, K);
    Z(1:K + 1:end) = 0.95;

end

