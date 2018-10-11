function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
    m = length(y); % number of training examples
    h = sigmoid(X * theta);
    J = sum(- y .* log(h) + (y - 1) .* log(1 - h)) / m;
    grad = X' * (h - y) ./ m;
end
