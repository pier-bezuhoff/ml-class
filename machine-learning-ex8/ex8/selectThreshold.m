function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
% [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
% threshold to use for selecting outliers based on the results from a
% validation set (pval) and the ground truth (yval).
  bestEpsilon = 0;
  bestF1 = 0;
  F1 = 0;

  stepsize = (max(pval) - min(pval)) / 1000;
  for epsilon = min(pval):stepsize:max(pval)
    ps = pval < epsilon;
    tp = nnz(ps & (ps == yval));
    fp = nnz(ps & (ps != yval));
    fn = nnz(!ps & (ps != yval));
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    F1 = 2 * prec * rec / (prec + rec); % 2 * tp / (2 * tp + fp + fn);
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    if F1 > bestF1
      bestF1 = F1;
      bestEpsilon = epsilon;
    end
  end
end
