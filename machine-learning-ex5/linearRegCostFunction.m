function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% hTheta() = (theta0 * x0) + (theta1 * x1) + ... + (theta_n * x_n)
% X = [ones(size(X,1) , 1) , X]; 

hTheta = X*theta; 

% hTheta = X(12X2) * theta(2X1) = hTheta(12X1) 
% y = y(12X1)
% (hTheta-y).^2 = I(12X1) = ((hTheta - y).^2)
% II = (theta(2:end).^2)
% As i use sum() function my 12X1 Matrix convert into a 1X1 matrix 
% I(1X1) + II(1X1)

J = (1/(2*m))*((hTheta - y)' * (hTheta - y)) + (lambda/(2*m))*(theta(2:end)' * theta(2:end)) ; 

% I (hTheta(12X1) - y(12X1)) * X(12X1);
% grad(1) = (1/m)*((hTheta - y))'*X(:, 1); % gradient(1) is (12X1)  
% grad(2:end) = (1/m)*((hTheta-y)'*X(:, 2:end))+((lambda/m) * theta(2:end))

thetaZero = theta; 
thetaZero(1) = 0;

% (htheta-y)' => ((1X12) * X(12X2))'' => (2X1) + (2X1) => 2X1 
grad = (1/m)*((hTheta-y)'*X)'+((lambda/m) * thetaZero)

% =========================================================================

grad = grad(:);

end
