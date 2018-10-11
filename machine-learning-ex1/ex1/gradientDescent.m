function [theta, Js] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
% theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha
% X[m x n], y[m x 1], theta[n x 1]
  m = length(y); % number of training examples
  Js = zeros(num_iters, 1);
  % ths = zeros(num_iters, 2);
  for iter = 1:num_iters
	% ths(iter, 1) = theta(1); ths(iter, 2) = theta(2);
	delta = 1 / m * ((X * theta - y)' * X);
	theta -= alpha * delta';
	Js(iter) = computeCost(X, y, theta); % Save the cost J in every iteration
  end
end
