%
% Prepare environment
%
if exist('OCTAVE_VERSION') ~= 0
    % pkg install -forge statistics
    pkg load statistics
    % pkg install -forge parallel
    pkg load parallel
    addpath(genpath('octave-additionals'))
else
    addpath(genpath('matlab-additionals'))
end

addpath(genpath('algorithm'))


rng('shuffle');
input_data_folder = 'data/FS_16ROI_mean/';
subject_filenames = {
	'6251_mean_fs_heavy'
	'6389_mean_fs_heavy'
	'6492_mean_fs_heavy'
	'6550_mean_fs_heavy'
	'6551_mean_fs_heavy'
	'6617_mean_fs_heavy'
	'6643_mean_fs_heavy'
	'6648_mean_fs_heavy'
	'6737_mean_fs_heavy'
	'6743_mean_fs_heavy'
	'6756_mean_fs_heavy'
	'6769_mean_fs_heavy'
	'6781_mean_fs_heavy'
	'6791_mean_fs_heavy'
};
output_figure_folder = './results/FS_16ROI_mean_heavy/figures/';
output_data_folder = './results/FS_16ROI_mean_heavy/data/';
%
input_data_folder = 'data/FS_16ROI_mean/';
subject_filenames = {
	'6251_mean_fs'
	'6389_mean_fs'
	'6492_mean_fs'
	'6550_mean_fs'
	'6551_mean_fs'
	'6617_mean_fs'
	'6643_mean_fs'
	'6648_mean_fs'
	'6737_mean_fs'
	'6743_mean_fs'
	'6756_mean_fs'
	'6769_mean_fs'
	'6781_mean_fs'
	'6791_mean_fs'
};
output_figure_folder = './results/FS_16ROI_mean/figures/';
output_data_folder = './results/FS_16ROI_mean/data/';
%
subject_filename = '6251_mean_fs';

filename = strcat(input_data_folder, subject_filename, '.mat');
kmindconnect.algorithm(filename)




