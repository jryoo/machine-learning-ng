%% Initialization
clear ; close all; clc

% Load the trained model from ex6_spam.m
load('spam_model.mat');
load('spamTrain.mat');
load('spamTest.mat');

%% =================== Part 6: Try Your Own Emails =====================
% spamSample1.txt

filenames = {'spamSample1.txt',
            'spamSample2.txt',
            'emailSample1.txt',
            'emailSample2.txt',
            'myEmailSample1.txt',
            'myEmailSample2.txt',
            'mySpamSample1.txt',
            'mySpamSample2.txt'};

for i=1:rows(filenames),
    % Read and predict
    name = filenames{i};
    file_contents = readFile(name);
    word_indices  = processEmail(file_contents);
    x             = emailFeatures(word_indices);
    p = svmPredict(model, x);

    fprintf('\nProcessed %s\n\nSpam Classification: %d\n', name, p);
    fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
end;

