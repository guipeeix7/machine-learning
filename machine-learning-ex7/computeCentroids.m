function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
% VARIABLES
% idx = idx(300X1)
% K = number of Centroids
% centroids = (3X2)
% END OF VARIABLES

size(centroids)

for i = 1:K
    % all(idx == i,2) = (300X1)
    % X = (300X2)
    % find(all(idx == i,2)); 
    matchedIndexes = find(all(idx == i , 2)); %get all row lines where idx equals 1 according to the acual number of centroid
    Ck = size(matchedIndexes,1); %Get the lenght of matched array above 
    %Also a little non-paid advertisment to Calvin Klein
    centroids(i,:) = (1/Ck)*(sum( X(matchedIndexes,:)));

    % The three line above also could been done with 
    % centroids(k, :) = mean(X(idx==k, :));

end





% =============================================================


end

