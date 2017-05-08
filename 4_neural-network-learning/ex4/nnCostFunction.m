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

%Add ones to the X data matrix
X = [ones(m, 1) X];
A = sigmoid(X*Theta1');

% Add ones to the A data matrix (hidden layer)
A = [ones(m, 1) A];
H = sigmoid(A*Theta2');

for i=1:num_labels,
J = J + 1/m*(-(y==i)' * log(H(:,i)) - (1-(y==i))' * log(1-H(:,i)));
end;

% apply regularisation
% ignore theta(1) to prevent regularising it
theta1_regularised = Theta1;
theta1_regularised(:,1) = 0;
theta2_regularised = Theta2;
theta2_regularised(:,1) = 0;

J = J + lambda / 2 / m * (sum(sumsq(theta1_regularised)) + sum(sumsq(theta2_regularised)));

% -------------------------------------------------------------
y_matrix = repmat(y, 1, num_labels) == repmat(1:num_labels, m, 1);
DELTA1 = zeros(hidden_layer_size,input_layer_size + 1);
DELTA2 = zeros(num_labels,hidden_layer_size + 1);

for t = 1:m,
a1_bias = X(t,:)';
a1 = a1_bias(2:end);
a2 = sigmoid(Theta1*a1_bias);
a2_bias = [1; a2];
a3 = sigmoid(Theta2*a2_bias);

% d3 is 10x1
% d2 is 26x1
d3 = a3 - y_matrix(t,:)';
d2 = (Theta2' * d3) .* a2_bias .* (1 - a2_bias);
%d2 = (Theta2' * d3) .* sigmoidGradient(Theta1*a1);

DELTA1 = DELTA1 + (d2(2:end) * a1_bias');
DELTA2 = DELTA2 + (d3 * a2_bias');
end;

Theta1_grad = DELTA1./m;
Theta2_grad = DELTA2./m;

%y_matrix = repmat(y, 1, num_labels) == repmat(1:num_labels, m, 1);
%delta3 = H - y_matrix;
%delta2 = Theta2' * delta3 .* sigmoidGradient(A*Theta2');

%DELTA2 = 0
%DELTA2 = DELTA2 + delta3 * H'

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
