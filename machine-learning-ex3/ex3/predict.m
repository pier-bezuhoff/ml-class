function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
    m = size(X, 1);
    X = [ones(m, 1) X];
    A2 = sigmoid(X * Theta1');
    A2 = [ones(m, 1) A2];
    [_ p] = max(sigmoid(A2 * Theta2'), [], 2);
end
