function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
% [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
% cost of using theta as the parameter for linear regression to fit the
% data points in X and y. Returns the cost in J and the gradient in grad
  m = length(y); % number of training examples
  J = sumsq(X * theta - y) / (2 * m) + sumsq(theta(2:end)) * lambda / (2 * m);
  grad = X' * (X * theta - y) / m + [0; theta(2:end)] * lambda / m;
end
