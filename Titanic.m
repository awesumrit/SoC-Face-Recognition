
clear ; close all; clc

load('data.txt');
X = data(:, [3,4,5,6,7]); y = data(:, 2);

[m, n] = size(X);

X = [ones(m, 1) X];


initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
m = size(X, 1); 

p = zeros(m, 1);

index = find(sigmoid(X*theta)>=0.525) ;
p(index) =1; 

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);




% test(1,:)= [];
% M=test ;
% M(:,[2,3,4,5,6,7,8,9,10,11,12])= [];
% test(:,[3,4,5,9,11,12])= [];
% X = test(:, [2,3,4,5,6]);
% [m, n] = size(X);

% Add intercept term to x and X_test
% X = [ones(m, 1) X];

% m = size(X, 1); 

% p = zeros(m, 1);

% index = find(sigmoid(X*theta)>=0.75) ;
% p(index) =1; 

