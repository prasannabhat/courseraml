function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

%% =================== Calculate theta by normal equation ===================
theta_normal = pinv(X' * X) * X' * y;
% print theta to screen
fprintf('Theta found by normal equation: ');
fprintf('%f %f \n', theta_normal(1), theta_normal(2));
normalCost = computeCost(X, y, theta_normal);
fprintf('Final cost by normal equation : %f\n',normalCost);


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
interval = 100;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	%Lets calculate constant factor
	% Compute hypothesis vector
	H = X * theta;
	%Error vector = hypothesis - actual
	E = H - y;
	delta = ((E' * X)')/ m; % compute the delta term
	theta = theta - (alpha * delta);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

figure; % open a new figure window
plot(J_history,'r'); % Plot the data
ylabel('Values of cost function'); % Set the y􀀀axis label
xlabel('Iterations'); % Set the x􀀀axis label

hold on;
plot(normalCost * ones(num_iters,1),'b');
%plot ([0,0], [1500,1500]);
legend('Cost Progress', 'Ideal cost');
hold off % don't overlay any more plots on this figure
%
end
%