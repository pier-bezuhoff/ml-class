function [J grad] = nnCostFunction(nn_params, ...
                                   n_inputs, ...
                                   n_hiddens, ...
                                   n_outputs, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, n_hiddens, n_outputs, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:n_hiddens * (n_inputs + 1)), ...
                     n_hiddens, (n_inputs + 1));
    Theta2 = reshape(nn_params((1 + (n_hiddens * (n_inputs + 1))):end), ...
                     n_outputs, (n_hiddens + 1));
    m = size(X, 1);
    a2 = sigmoid([ones(m, 1) X] * Theta1');
    h = sigmoid([ones(m, 1) a2] * Theta2');
    Y = zeros(m, n_outputs); % label -> orth vector
    for i = 1:m
        Y(i, y(i)) = 1;
    end
    sq1 = sumsq(Theta1(:, 2:end)(:));
    sq2 = sumsq(Theta2(:, 2:end)(:));
    J = -sum(sum(Y .* log(h) + (1 - Y) .* log(1 - h))) / m + ...
        lambda * (sq1 + sq2) / (2 * m); % regularization
    a1 = [ones(m, 1) X]; % m x (n_inputs + 1) 
    z2 = a1 * Theta1'; % m x n_hiddens
    a2 = [ones(m, 1) sigmoid(z2)]; % m x (n_hiddens + 1)
    z3 = a2 * Theta2'; % m x n_outputs
    a3 = sigmoid(z3); % m x n_outputs
    delta3 = a3 - Y; % delta3: m x n_outputs, delta2: m x n_hiddens
    delta2 = (delta3 * Theta2(:, 2:end)) .* a2(:, 2:end) .* (1 - a2(:, 2:end)); % sigmoidGradient(z2);
    Delta1 = delta2' * a1; % n_hiddens x (n_inputs + 1)
    Delta2 = delta3' * a2; % n_outputs x (n_hiddens + 1)
    Theta1_grad = Delta1 ./ m;
    Theta1_grad(:, 2:end) += lambda / m * Theta1(:, 2:end);
    Theta2_grad = Delta2 ./ m;
    Theta2_grad(:, 2:end) += lambda / m * Theta2(:, 2:end);
    grad = [Theta1_grad(:); Theta2_grad(:)];
end
