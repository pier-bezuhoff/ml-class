function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
  % theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
  % taking num_iters gradient steps with learning rate alpha
  % X[m x n], y[m x 1], theta[n x 1]
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
	delta = 1 / m * ((X * theta - y)' * X);
	theta -= alpha * delta';
	J_history(iter) = computeCost(X, y, theta); % Save the cost J in every iteration
  end
end
