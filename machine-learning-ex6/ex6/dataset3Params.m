function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
% [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
% sigma. You should complete this function to return the optimal C and 
% sigma based on a cross-validation set.
  # interval = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
  # error = @(C, sigma) mean(double(svmPredict(svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)), Xval) ~= yval));
  # [X Y] = meshgrid(interval, interval);
  # Z = arrayfun(error, X, Y);
  # [minError i] = min(Z(:))
  # C = X(:)(i)
  # sigma = Y(:)(i)
  # figure(2);
  # surf(X, Y, Z);
  # xlabel("C"); ylabel("\sigma"); zlabel("error");
  # figure(1);
  # pause;
  C = 1;
  sigma = 0.1;
end
