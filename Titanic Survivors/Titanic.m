


load('data.txt');
X = data(:, [3,4,5,6,7,8]); y = data(:, 2);

[m, n] = size(X);

X = [ones(m, 1) X];

test_theta = zeros(n+1,1);
[cost, grad] = costFunction(test_theta, X, y );

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad(1:5));

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunction(t, X, y)), test_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
m = size(X, 1); 

p = zeros(m, 1);

index = find(sigmoid(X*theta)>=0.5) ;
p(index) =1; 

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

