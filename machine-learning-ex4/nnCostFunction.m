function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%KIND WORKS HERE (:'
% X = [ones(m, 1) X];

 
% a1 = sigmoid(X*Theta1');
% a1 = [ones(size(a1,1), 1) a1]; %Works like a piece of cake adding one extra column of ones into every training set
% a2 = sigmoid(a1*Theta2'); 
% yComp = zeros(m, 10);

% % This makes my life esiest to do the multiplication with a2 that turn a y matrix 5000x1 into 5000x10 (thats the number of classes)
% for i = 1:m
%     yComp(i, y(i)) = 1;
% endfor

% J = (1/m)*((-yComp.*log(a2)) - ((1-yComp).*log(1-a2))); %Thats outputs me a 5000x10 matrix
% J = sum(sum(J)); %This summ all lines and collums and gives me my final J
%END OF THAT

X = [ones(m,1) X];

a1 = X; 

z2 = (a1*Theta1');
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2]; 

z3 = (a2*Theta2');
a3 = sigmoid(z3);

h_theta = a3; 

yComp = zeros(m, num_labels);
for i = 1:m
    yComp(i, y(i)) = 1;
endfor

J = 1/m*(-yComp.*log(h_theta) - ((1 - yComp) .* (log(1-h_theta)))); 
J = sum(sum(J));

%Theta without the bias unit
thetaWtFs1 = Theta1(:, 2:end);
thetaWtFs2 = Theta2(:, 2:end);



sump1 = sum(sum(thetaWtFs1.*thetaWtFs1));
sump2 = sum(sum(thetaWtFs2.*thetaWtFs2));

regularization = (lambda/(2*m))*(sump1+sump2);

J = J+regularization; 
% return; 

% return; 
% for i = 1:10 
%     J = J + (1/m)*((-y'*log(a2(:, i))) - ((1-y)'*log(1-a2(:,i))));
% endfor
% size(J);
% sigX_frs_layer = sigmoid(X*Theta1'); 
% sigX_scd_layer = sigmoid(X*Theta2');

% J = (1/m)*(-y*log(sigX_frs_layer) - (1-y) * log(1 - sigX_frs_layer) ) 
% J = (1/m)*(-y*log(sigX_scd_layer) - (1-y) * log(1 - sigX_scd_layer) ) 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% delta3 = (a3 - yComp); 



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


for i = 1:m 
    %BEGIN OF FOWARD PROPAGATION
        a1 = X(i,:); %a1 = a1(1X401)
        
        z2 = a1*Theta1'; %z = a1(1x401)*Theta1(25X401)
        %z2 = a1(1X401)*Theta(401X25)' => z(1X25)
        a2 = sigmoid(z2);  %a2 = a2(1X25)
        a2 = [1 a2]; %a2 => a2(1X26) 

        z3 = a2*Theta2'; %z3 = a2(1X26) * Theta2(26X10) => z3 = z3(1X10)
        a3 = sigmoid(z3); 
    %END OF FOWARD PROPAGATION
    
    %BEGIN OF BACKWARD PROPAGATION
        delta3 = (a3-yComp(i)); %sigma3 = a3(1X10) - yComp(1X10) => sigma3 = sigma3(1X10)
        
        delta2 = (Theta2'*sigma3').*(sigmoidGradient([1 z2])'); 
                %(Theta2(26X10)'*sigma3(1*10)).*g'(z2)(1X26)
                %(Theta2(26X10)'*sigma3(10*1)).*g'(z2)(1X26)
                %(Theta2(26X10)'*sigma3(10*1)).*g'(z2)(26X1)'
                %sigma2 = sigma2(26X1)
                %The trick here is to invert the a1 matrix to the formula be equal to pdf 
        % delta2 = delta2(2:end);  
        

        delta = delta+sigma
    %END OF BACKWARD PROPAGATION

endfor















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
