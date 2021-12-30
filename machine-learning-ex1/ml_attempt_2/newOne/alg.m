clear ; close all; clc
fprintf('Loading data ...\n');
data = load('dataset.txt');
X = data(:, [1:2]);

y = data(:, 3);
m = length(y);

% Choose some alpha value
alpha = 0.01;
num_iters = 10000;
theta = zeros(3, 1);

J_history = zeros(num_iters, 1);
nThetas = length(theta);

% mu = mean(X(:,2));
% sigmaMax = max(X(:,2));
% sigmaMin = min(X(:,2));

% X_norm = ((X(:,2).-mu)./(sigmaMax-sigmaMin));
% X(:,2) = X_norm(:,2); 
mu = mean(X); %mean is the same as media
sigma = std(X);

X_norm = ((X.-mu)./(sigma));

% X_norm(:,1) = X_norm(:,1)/1000;

X = X_norm;

% fprintf('%f  %f  %f\n', X(:, 1), X(:,2) , X(:, 3));
X = [ones(m, 1) X];
% X(:,3)
% for iter = 1:num_iters
%     theta1 = theta(1) - ((alpha/m)*sum(((X*theta)-y)));
%     theta2 = theta(2) - ((alpha/m)*sum(((X*theta)-y)'*X(:,2)));
%     theta3 = theta(3) - ((alpha/m)*sum(((X*theta)-y)'*X(:,3)));
%     theta(1) = theta1;  
%     theta(2) = theta2;
%     theta(3) = theta3;
%     % J_history(iter) = computeCostMulti(X, y, theta);
% end  
auxTheta = zeros(nThetas, 1);
for iter = 1:num_iters
    for i = 1:nThetas
        auxTheta(i) = theta(i) - ((alpha/m)*sum(((X*theta)-y)'*X(:,i)));
    end
    theta = auxTheta;
    J_history(iter) = computeCostMulti(X, y, theta);
end
price = 0; % You should change this
theta
% price = theta(1)+(theta(2)*1650)+(theta(3)*3)

house = (([1650, 2].-mu)./(sigma));
% rooms = ((2.-mu)./(sigma));

price = theta(1)+(theta(2)*house(1))+(theta(3)*house(2))
% price = theta(1)+(theta(2)*5.8)
% fprintf(' %f \n', theta);
% fprintf('\n');

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

%    3.4041e+05
%    1.0964e+05
%   -6.5900e+03