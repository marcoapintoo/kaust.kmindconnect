%-------------------------------------------------------------------------%
%            Initialized by K-mean Clustering of TV-VAR coeffs            %
%-------------------------------------------------------------------------%
% p: VAR model order
% K: Number of states
% wlen: Window length
% shift: window shift (local)
% min_r_len: minimum length for each regime (local)
% y: signal y

function [St_km, A_km, C] = cluster_data(p, K, tvvar_vec, y)

    [St_km, C, ~, ~] = kmindconnect.clustering.variable_neighbour_search(tvvar_vec, K);

    [N, T] = size(y);
    % Pooling samples for regimes
    St = zeros(N, T, K);
    tj = zeros(K, 1);
    for j = 1:K
        t = 1;
        for i = 1:T
            if St_km(i) == j
                St(:, t, j) = y(:, i);
                t = t + 1;
            end
        end
        tj(j) = t - 1;
    end

    % Estimate state-specific VAR
    A_km = zeros(N, N * p, K);
    for j = 1:K
        [A_km(:, :, j), ~] = varfit(p, St(:, 1:tj(j), j));
    end

end
