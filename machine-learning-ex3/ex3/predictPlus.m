function [p stat] = predictPlus(Theta1, Theta2, X)
% return p [n_examples x 1] and stat [n_examples x n_classes], where
% p contains most likely classes and stat -- probabilitis for every class
    X = [ones(size(X, 1), 1) X];
    A2 = sigmoid(X * Theta1');
    A2 = [ones(size(A2, 1), 1) A2];
    A3 = sigmoid(A2 * Theta2');
    [_ p] = max(A3, [], 2);
    stat = A3;
end
