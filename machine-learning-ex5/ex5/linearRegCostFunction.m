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


J = 1 / (2 * m) * sum(((X * theta) - y) .^ 2) + (lambda / ( 2 * m)) * sum(theta(2:end) .^ 2);

% When we are calculating gradients,
% we need to calculate every gradient with the index i in our gradient vector
% according to its corresponding column i in our data matrix (X in this case).


% Let's take below vector with the original values from X
% into consideration

%    X matrix      theta vector
% [ 1 -15.9368 ]   [ thete_1 ]
% [ 1 -29.1540 ]   [ theta_2 ]
% [ 1 36.1895  ]
% [ 1 37.4922  ]
%
% Since we do not regularize theta_0
% gradient(1) will be sum of the column vector
% evaluated by (X * theta) - y times the actual values of the corresponding column from matrix X, namely X(1).
% Which is the first column of X in this case.

grad(1) = (1 / m) * sum((X * theta) - y .* X(:,1), 1);

% We use regularization for the rest of our gradient calculations.
% We just add the transpose of (lambda / m) times our theta values starting from 2nd index
% to calculate sum of each column in our 

grad(2:end) = (1 / m) * sum(((X * theta) - y) .* X(:, 2:end), 1) + ((lambda / m) * theta(2:end))'; 

% [ sum(((X * theta) - y) .* X(:, 2:end))(1)	sum(((X * theta) - y) .* X(:, 2:end))(2)	... ]
% [ theta(2)									theta(3)									... ]
% ------------------------------------------------------------------------------------------------ +
%

% =========================================================================

grad = grad(:);

end
