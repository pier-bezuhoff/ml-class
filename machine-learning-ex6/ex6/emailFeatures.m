function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
% x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
% produces a feature vector from the word indices. 
  n = 1899; % Total number of words in the dictionary
  x = arrayfun(@(n) nnz(word_indices == n) > 0, 1:n);
end
