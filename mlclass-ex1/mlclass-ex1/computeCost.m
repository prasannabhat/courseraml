function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% X is a m * 2 matrix
% Compute hypothesis vector
H = X * theta;
%Error vector = hypothesis - actual
E = H - y;
E = E .^ 2; %Square the error
J = sum(E)/(2 * m);





% =========================================================================

end
