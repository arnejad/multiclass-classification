function [preds] = logisticRegression(X_train, X_test, y_train, y_test, K)
%LOGISTICREGRESSION operates a multinomial Logistic Regression on input

%% INITIALIZATIONS
lambda = 0.01;  %regulization coef
epochs = 300;
[N,D] = size(X_train);
W = rand(D, K);
b = zeros(1, K);
l2 = 0.0;
epochs_costs = [];

% converting labels to one-hot represatiation
y_hot = full(ind2vec(y_train',K))';

%% TRAIN
% Gradient Descent
for e=1:epochs
    
    diff = softmax((X_train * W + b)')' - y_hot; % calculating gradient
    grad = X_train' * diff;
    W = W - (lambda * grad + (lambda * l2 * W)); % Updating W & b
    b = b - (lambda * sum(diff));    
end

%% TEST

sm = softmax((X_test * W + b)')';
[~ , preds] = max(sm, [], 2);

end

