function [A0, H0, Q0, R0, x_0] = initialize_svar(p, K, A_km, y)
    [N, ~] = size(y);

    A0 = zeros(N * p, N * p, K);
    H0 = zeros(N, N * p, K);
    Q0 = zeros(N * p, N * p, K);
    R0 = zeros(N, N, K);
    x_0 = zeros(N * p, 1);

    % Initialize SVAR parameters
    for j = 1:K
        A0(1:N, :, j) = A_km(:, :, j);
        for k = 1:p
            if k < p
                A0(k * N + 1:k * N + N, (k - 1) * N + 1:(k - 1) * N + N, j) = eye(N, N);
            end
        end
        Q0(1:N, 1:N, j) = eye(N);
        H0(1:N, 1:N, j) = eye(N);
        R0(:, :, j) = 0.1 * eye(N);
    end

end