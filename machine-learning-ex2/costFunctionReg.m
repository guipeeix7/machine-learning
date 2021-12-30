function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hip = sigmoid(X*theta);

size(y);
size(hip);
size(theta);

%theta = theta
thetaToCost = theta(2:size(theta)(1));%Not regularizing the 0 index 

J = ((1/m)*sum( (-y' * log(hip)) - (1-y')*log(1-hip))) + ((lambda/(2*m)) * sum(thetaToCost.^2));

nFeatures = size(X)(2);
grad(1) = ((1/m) * sum((hip-y)' * X(:,1)));

for j = 2:nFeatures
    grad(j) = ((1/m) * sum((hip-y)' * X(:,j))) + ((lambda/m)*(theta(j))); %Using the normal theta array
endfor
% =============================================================

end
