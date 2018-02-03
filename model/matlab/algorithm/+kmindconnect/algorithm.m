%==========================================================================
%  Estimating dynamic connectivity states in movie fMRI
%
%  8-9-2017 Chee-Ming Ting
%  - switching VAR
%  - 16 Free-surfer pacellated ROIs
%==========================================================================

%-------------------------------------------------------------------------%
%                       Single-subject Analysis                           %
%-------------------------------------------------------------------------%
function algorithm(filename)
    % Algorithm parameters:
    p = 1; % VAR model order
    K = 3; % Number of states
    wlen = 50; % Window length
    shift = 1; % Window shift
    min_r_len = 5; % Minimum length for each regime

    eps = 0.0001; % Tolerated error in Markov Matrices
    ItrNo = 15; % Iteration number in Markov Matrices

    load(filename); % load data
    y = mean_roi;
    tvvar_vec = kmindconnect.time_variable_var(p, wlen, shift, y);
    [St_km, A_km, C] = kmindconnect.cluster_data(p, K, tvvar_vec, y);
    
    %time_varying_var_coefficients = tvvar_vec;
    %save(sprintf('Test_%s_01', codename), 'time_varying_var_coefficients', '-v6')
    %cluster_centres = C;
    %clustered_coefficients = St_km;
    %expanded_time_series = St;
    %length_by_cluster = tj;
    %time_varying_states_var_coefficients = A_km;
    %save(sprintf('Test_%s_02', codename), 'cluster_centres', 'clustered_coefficients', 'expanded_time_series', 'length_by_cluster', 'time_varying_states_var_coefficients', '-v6')

    fprintf(':: Applying Kalman filter...'); t0 = cputime;
    [A0, Q0, R0, fSt, sSt, Fxt, Sxt, Ahat, Qhat, Rhat, Zhat, Ls, St_skf, St_sks] = kmindconnect.switching_kalman_filter(p, K, eps, ItrNo, A_km, y);
    fprintf(' [OK: %.3fsecs]\n', cputime - t0);

end
 
 