%-------------------------------------------------------------------------%
%       Simple resemble of a Variable Neighbour Search using K-means      %
%-------------------------------------------------------------------------%
	
function [St_km, C, sumD, d] = variable_neighbour_search(tvvar_vec, K)
	[~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqeuclidean','Replicates',20);
	[~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqeuclidean','Start',C);
	[~, C, ~, d] = kmeans(tvvar_vec',K,'Distance','cityblock','Start',C);
	[~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqeuclidean','Start',C);
	[St_km, C, sumD, d] = kmeans(tvvar_vec',K,'Distance','cityblock','Start',C);
end