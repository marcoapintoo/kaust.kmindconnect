%-------------------------------------------------------------------------%
%                  Plot State-dependent VAR coeff matrix  (K-means)       %
%-------------------------------------------------------------------------%
%N:
%p: 
%K:
%A_km:
%subject_filename:
%figure_folder:

function svar_coeff_matrix(N, p, K, Ahat, subject_filename, figure_folder)
    Channel_labelsY = {'CG.L', 'PP.L', 'PT.L', 'PoCG.L', 'PreCG.L', 'STG.L', 'TTG.L', 'TTS.L', 'CG.R', 'PP.R', 'PT.R', 'PoCG.R', 'PreCG.R', 'STG.R', 'TTG.R', 'TTS.R'};
    Channel_labelsX = {'CG.L', 'PP.L', 'PT.L', 'PoCG.L', 'PreCG.L', 'STG.L', 'TTG.L', 'TTS.L', 'CG.R', 'PP.R', 'PT.R', 'PoCG.R', 'PreCG.R', 'STG.R', 'TTG.R', 'TTS.R'};

    % SVAR
    for j = 1:K
        figureName = sprintf('FS-svar-VARmat-p%dK%d-state%d-sub%s', p, K, j, subject_filename);
        figure('Name', figureName, 'Color', [1 1 1]);
        imagesc(squeeze(Ahat(:, 1:N * p, j)));
        colormap(jet);
        h = colorbar;
        axis square;
        caxis([min(Ahat(:)) max(Ahat(:))]);
        ylabel('ROIs', 'FontSize', 18, 'fontweight', 'bold');
        xlabel('ROIs', 'FontSize', 18, 'fontweight', 'bold');
        set(gca, 'XTick', 1:1:N, 'XTickLabel', Channel_labelsX);
        set(gca, 'YTick', 1:1:N, 'YTickLabel', Channel_labelsY);
        set(gca, 'FontSize', 12);
        set(h, 'fontsize', 12);
        svFigName = strcat(figure_folder, figureName, '.eps');
        saveas(gcf, svFigName, 'epsc2');
    end