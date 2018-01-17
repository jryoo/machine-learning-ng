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

X = [ones(m,1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feed forward
% 401 --> 25 --> 10
z2 = X*Theta1';
a2 = sigmoid(z2);
% add the bias term
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

% Convert y to boolean and create columns for classifications
y_mask = repmat(1:size(a3,2),size(y,1),1);
y_complete = repmat(y,1,size(a3,2));
y_complete = y_mask == y_complete;

% fprintf(['nnCostFunction: Feed Forward\n']);
% fprintf(['Size of X: %d x %d \n'], size(X));
% fprintf(['Size of z2: %d x %d \n'], size(z2));
% fprintf(['Size of Theta1: %d x %d \n'], size(Theta1));
% fprintf(['Size of a2+bias: %d x %d \n'], size(a2));
% fprintf(['Size of z3: %d x %d \n'], size(z3));
% fprintf(['Size of Theta2: %d x %d \n'], size(Theta2));
% fprintf(['Size of a3 (h): %d x %d \n'], size(a3));
% fprintf(['Size of y_complete: %d x %d \n'], size(y_complete));

J = (1/m) * sum(sum((-y_complete.*log(a3)) - (1-y_complete).*log(1-a3),2));

% Regularization

% The first column is the bias, which we do not include
reg_theta1 = Theta1;
reg_theta1(:,1) = 0;
reg_theta2 = Theta2;
reg_theta2(:,1) = 0;

regularization = (lambda/(2*m)) .* (sum(sum((reg_theta1.^2))) + sum(sum(reg_theta2.^2)));
J = J + regularization;


% Backpropagation
% fprintf(['nnCostFunction: Backpropagation\n']);
% fprintf(['Size of X: %d x %d \n'], size(X));
% fprintf(['Size of Theta1_grad: %d x %d \n'], size(Theta1_grad));
% fprintf(['Size of Theta2_grad: %d x %d \n'], size(Theta2_grad));

% Iterative Algorithm
% for i=1:m,
%     x_i = X(i,:);
%     z_2 = Theta1*x_i';
%     a_2 = sigmoid(z_2);
%     a_2 = [1;a_2];
%     z_3 = Theta2*a_2;
%     a_3 = sigmoid(z_3);
%     delta_3 = a_3 - y_complete(i,:)';
%     % Note that delta(2)_0 is not used
%     delta_2 = (Theta2'*delta_3).*[1;sigmoidGradient(z_2)];
%     Theta1_grad = Theta1_grad + delta_2(2:end)*(x_i);
%     Theta2_grad = Theta2_grad + delta_3*a_2';
% end;

% Vectorized Algorithm
delta_3 = a3 - y_complete;
% Note that delta(2)_0 is not used
delta_2 = (delta_3*Theta2).*[ones(rows(z2),1), sigmoidGradient(z2)];
Theta1_grad = Theta1_grad + delta_2(:,2:end)'*(X);
Theta2_grad = Theta2_grad + delta_3'*a2;

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularization
Theta1_grad = Theta1_grad.+[zeros(rows(Theta1),1) ((lambda/m).*Theta1(:,2:end))];
Theta2_grad = Theta2_grad.+[zeros(rows(Theta2),1) ((lambda/m).*Theta2(:,2:end))];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
