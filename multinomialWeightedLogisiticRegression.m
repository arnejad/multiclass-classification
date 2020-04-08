function [preds] = multinomialWeightedLogisiticRegression(X_train, X_test, y_train, y_test, K)	
%MULTINOMIALWEIGHTEDLOGISITICREGRESSION 

%% INITIALIZATIONS
lambda = 0.01;  %regulization coef
PARAM = 0.5;
epochs = 500;
[N,D] = size(X_train);
l2 = 0.0;       %learning rate
epochs_costs = [];

% transforming  labels to one-hot representation
y_hot = full(ind2vec(y_train',K))';

%% TRAIN & TEST

preds = [];
for t=1:size(X_test,1)  %for each test data
    
    %calculate weights
    W = exp(-(((dist(X_test(t,:), X_train').^2)/(2 * (PARAM)^2))))';
    
    %initialing coef parameters
    theta = rand(D, K);
    b = zeros(1, K);
    
    %Gradient Descent
    for e=1:epochs
        diff = softmax(((X_train.*W) * theta + b)')' - y_hot;
%         grad = (X_train.*W)' * diff;
        grad = (X_train)' * diff;
        theta = theta - (lambda * grad + (lambda * l2 * theta));
        b = b - (lambda * sum(diff));
    end
    
    sm = softmax((X_test(t,:) * theta + b)')';
    [~ , tempPred] = max(sm, [], 2);
    fprintf('Test data number %d predicted\n', t);
    preds = [preds; tempPred];
end
%% TEST
% sm = softmax((X_test * theta + b)')';
% [~ , preds] = max(sm, [], 2);

%% Accuracy metrics calculation
% corrects = 0;
% for i=1:length(preds)
%     if preds(i) == y_test(i)
%         corrects = corrects + 1;
%     end
% end
% 
% fprintf('Number of correct preds: %d', corrects)

end

