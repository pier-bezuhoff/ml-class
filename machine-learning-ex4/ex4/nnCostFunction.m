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
    last = n_hiddens * (n_inputs + 1);
    nn_params(1:n_hiddens) = 0; % do not count biases
    step = n_hiddens * (n_hiddens + 1);
    till_end = n_outputs * (n_hiddens + 1);
    for i = 1:n_hiddens
        nn_params(last+i:step:end-till_end-1) = 0;
    end
    % TODO: check zero-ing hidden layers biases
    nn_params(end-till_end:end-till_end+n_outputs) = 0;
    J = -sum(sum(Y .* log(h) + (1 - Y) .* log(1 - h))) / m + ...
        lambda * sumsq(nn_params) / (2 * m); % regularization
    a1 = X; % m x n_inputs
    z2 = [ones(m, 1) X] * Theta1'; % m x n_hiddens
    a2 = sigmoid(z2); % m x n_hiddens
    z3 = [ones(m, 1) z2] * Theta2'; % m x n_outputs
    a3 = sigmoid(z3); % m x n_outputs
    delta3 = a3 - Y; % delta3: m x n_outputs, delta2: m x n_hiddens
    delta2 = (delta3 * Theta2(:,2:end)) .* a2 .* (1 - a2); % sigmoidGradient(z2);
    Delta1 = delta2' * a1;
    Delta2 = delta3' * a2;
    Theta1_grad = Delta1 ./ m + lambda * Theta1(:,2:end);
    Theta2_grad = Delta2 ./ m + lambda * Theta2(:,2:end);
    % Part 3: Implement regularization with the cost function and gradients.
    %
    %         Hint: You can implement this around the code for
    %               backpropagation. That is, you can compute the gradients for
    %               the regularization separately and then add them to Theta1_grad
    %               and Theta2_grad from Part 2.
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end