%-------------------------------------------------------------------------%
%                        Plot State SVAR sequence                         %
%-------------------------------------------------------------------------%
%p: 
%K:
%T:
%St_km:
%subject_filename:
%figure_folder:

function svar_sequence(p, K, T, St_sks, subject_filename, figure_folder)
    t = 1:1:T;
    figureName = sprintf('FS-svar-states-p%dK%d-sub%s', p, K, subject_filename);
    figure('Name', figureName, 'Color', [1 1 1]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperPosition', [0 0 7 2]);
    plot(t, St_sks, 'Color', [0 102 204] / 255, 'LineWidth', 2);
    hold on;
    xlim([1 T]);
    ylim([.75 K + .25]);
    set(gca, 'XTick', 100:100:T, 'fontsize', 11);
    set(gca, 'YTick', 1:1:K, 'fontsize', 11);
    xlabel('Time Point', 'fontsize', 12);
    ylabel('States', 'fontsize', 12);
    svFigName = strcat(figure_folder, figureName, '.eps');
    saveas(gcf, svFigName, 'epsc2');
end
