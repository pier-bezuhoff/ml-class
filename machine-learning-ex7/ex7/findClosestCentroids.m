function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
% idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
% in idx for a dataset X where each row is a single example. idx = m x 1 
% vector of centroid assignments (i.e. each entry in range [1..K])
  K = size(centroids, 1);
  m = size(X, 1);
  %% my mid. perp. enhancement
  % vector of middle perpendicular hyperplane of 2 covectors
  midPerp = @(a, b) [b - a, - (b - a) * mean([a; b])']';
  midPerps = cell(K, K);
  for i = 1:K
    for j = i:K
      midPerps{i,j} = midPerp(centroids(i,:), centroids(j,:));
      midPerps{j,i} = -midPerps{i,j};
    end
  end
  idx = zeros(m, 1);
  for p = 1:m
    closeness = cellfun(@(perp) [X(p,:) 1] * perp >= 0, midPerps);
    idx(p) = find(all(closeness, 1));
  end
  %% and it does work!
  %% but sometimes strange conformance bug?!
  % idx = zeros(m, 1);
  % for i = 1:m
  %   [_ idx(i)] = min(sumsq(X(i,:) - centroids, 2));
  % end
end
