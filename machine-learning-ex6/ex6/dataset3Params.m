function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

range = 6;
start_value = 0.01;
multiplier = 3;
v_sets = {};

for i=1:range,
    for j=1:range,
        v_sets(end + 1) = [start_value * multiplier^i,
                            start_value * multiplier^j];
    end;
end;

errors = [];

for i=1:columns(v_sets),
    C = v_sets{1,i}(1);
    sigma = v_sets{1,i}(2);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    errors(i,1) =error;
end;

[val, row] = min(errors);

C = v_sets{1,row}(1);
sigma = v_sets{1,row}(2);

#C =  0.27000
#sigma =  0.09000

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
