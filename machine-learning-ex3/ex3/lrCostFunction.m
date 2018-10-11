function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
    m = length(y); % number of training examples
    h = sigmoid(X * theta);
    ths = theta(2:end); % for regularization
    J = sum(- y .* log(h) + (y - 1) .* log(1 - h)) / m + ...
        ths' * ths * lambda / (2 * m);
    grad = (X' * (h - y) + lambda * [0; ths]) ./ m;
end
