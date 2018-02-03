%-------------------------------------------------------------------------%
%            Initialized by K-mean Clustering of TV-VAR coeffs            %
%-------------------------------------------------------------------------%
% p: VAR model order
% K: Number of states
% wlen: Window length
% shift: window shift (local)
% min_r_len: minimum length for each regime (local)
% y: signal y

function [tvvar_vec] = time_variable_var(p, wlen, shift, y)
    [N, T] = size(y);

    tvvar_vec = zeros(p * N ^ 2, T); %TV-VAR coeffs
    win = rectwin(wlen); % form a window

    % initialize the indexes
    indx = 0; t = 1;
    Yw = zeros(N, wlen);

    % Short-Time VAR Analysis
    while indx + wlen <= T
        % windowing
        for i = 1:N
            Yw(i, :) = y(i, indx + 1:indx + wlen) .* win';
        end
        [At, ~] = varfit(p, Yw); % Fit a VAR
        tvvar_vec(:, t) = At(:); % update time-varying matrix
        indx = indx + shift;
        t = t + 1; % update the indexes
    end
end
