function unit = contour2DLR(X, y, ths, th00, th01, th10, th11, dots)
  % 2D contour plot of gradient descent over linear regression
  % design matrix X[m x 2], example results y[m x 1],
  % gradient history of theta[2 x 1] ths[N x 2],
  % gradient history of J Js[N x 1],
  % theta0 <- [th00; th01] (dots values),
  % theta1 <- [th10; th11] (dots values)
  th0s = linspace(th00, th01, dots);
  th1s = linspace(th10, th11, dots);
  Jss = zeros(dots, dots);
  for i = 1:dots
	for j = 1:dots
	  Jss(i, j) = computeCost(X, y, [th0s(i); th1s(j)]);
	end
  end
  contour(th0s, th1s, Jss, logspace(-2, 3, 20))
  xlabel('\theta_0'); ylabel('\theta_1');
  hold on;
  plot(ths(:,1), ths(:,2), "-r")
  unit = 1;
