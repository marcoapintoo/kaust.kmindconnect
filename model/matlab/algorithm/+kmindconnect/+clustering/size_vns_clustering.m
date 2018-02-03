%-------------------------------------------------------------------------%
%       Variable Neighbour Search using K-means using pre-clustering      %
%-------------------------------------------------------------------------%
	
function [St_km, C, sumD, d] = size_vns_clustering(tvvar_vec, K)
	[dim, rows] = size(tvvar_vec);
	max_rows = 4000;
	if rows < max_rows
		fprintf('[%d instances]', rows);
		% If the matrix is small enough, continue with VNS
		[St_km, C, sumD, d] = kmindconnect.clustering.variable_neighbour_search(tvvar_vec, K);
	else
		fprintf('[warmup: %d->%d instances]', rows, max_rows);
		% Created a sampled version of tvvar_vec, and clustering over it
		selected_rows = randi([1 rows], 1, max_rows);
		sampled_data = tvvar_vec(:, selected_rows);
		[St_km, C, sumD, d] = kmindconnect.clustering.variable_neighbour_search(sampled_data, K);

		fprintf('[clustering: %d instances]', rows);
		% Later, use that center as start center points for clustering the whole matrix 
		[~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqEuclidean','start',C);
		[~, C, ~, d] = kmeans(tvvar_vec',K,'Distance','cityblock','start',C);
		[~, C, ~, ~] = kmeans(tvvar_vec',K,'Distance','sqEuclidean','start',C);
		[St_km, C, sumD, d] = kmeans(tvvar_vec',K,'Distance','cityblock','start',C);
	end
end