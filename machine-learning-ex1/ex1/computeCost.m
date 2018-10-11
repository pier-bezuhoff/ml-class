function J = computeCost(X, y, theta)
  %COMPUTECOST Compute cost for linear regression
  % J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
  % parameter for linear regression to fit the data points in X and y
  % X[m x n], y[m x 1], theta[n x 1]
  m = length(y); % number of training examples
  J = sum((X * theta - y) .^ 2) / (2 * m);
end
