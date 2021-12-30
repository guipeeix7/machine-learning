function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
nThetas = length(theta)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % theta1 = theta(1) - ((alpha/m)*sum(((X*theta)-y)));
    % theta2 = theta(2) - ((alpha/m)*sum(((X*theta)-y)'*X(:,2)));
    % theta(1) = theta1;  
    % theta(2) = theta2;  

    % theta = theta - ((alpha/m)*((X*theta)-y)'*X)'; %This is kind the optimal, but i dont understand :c
    auxTheta = zeros(nThetas, 1);
    for i = 1:nThetas
        auxTheta(i) = theta(i) - ((alpha/m)*sum(((X*theta)-y)'*X(:,i)));
    end
    theta = auxTheta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end