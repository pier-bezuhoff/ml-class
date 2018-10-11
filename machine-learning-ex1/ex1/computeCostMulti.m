function J = computeCostMulti(X, y, theta)
  %COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
  % J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
  % parameter for linear regression to fit the data points in X and y
  % X[m x n], y[m x 1], theta[n x 1]
  m = length(y); % number of training examples
  errors = X * theta - y;
  J = errors' * errors / (2 * m);
  % J = sum((X * theta - y) .^ 2) / (2 * m);
end
