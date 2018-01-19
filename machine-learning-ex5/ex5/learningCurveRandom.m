function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
random_samples = 50;

for i=1:m,
    error_train_i = 0;
    error_val_i = 0;
    for j=1:random_samples,
        rndIDX = randperm(m);
        rnd_ts_x = X(rndIDX(1:i),:);
        rnd_ts_y = y(rndIDX(1:i));
        rnd_val_x = Xval(rndIDX(1:i),:);
        rnd_val_y = yval(rndIDX(1:i));
        ts_theta = trainLinearReg(rnd_ts_x, rnd_ts_y, lambda);
        error_train_i = error_train_i + linearRegCostFunction(rnd_ts_x, rnd_ts_y, ts_theta, 0);
        error_val_i = error_val_i + linearRegCostFunction(rnd_val_x, rnd_val_y, ts_theta, 0);
    end
    error_train(i,1) = error_train_i / random_samples;
    error_val(i,1) = error_val_i / random_samples;
end;

% You need to return these values correctly
% error_train = zeros(m, 1);
% error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------







% -------------------------------------------------------------

% =========================================================================

end
