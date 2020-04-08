function [probs] = binaryBayesianLogReg(X_train, X_test, y_train, y_test, K)
%BAYESIANLOGREG implements Bayesian Logistic Regression for binary methods

%Reference: Bishop, C. M. (2006). Pattern recognition and machine learning. 
            %Springer (Chapter 4.5.1)
    
            
% method   for Newton Rhapson method 'NR' and for Gradient Descent 'GD'

%% INITIALIZATION
[N,D] = size(X_train);
method = 'GD';

prior_variance = 100 * eye(D);
prior_mean = zeros(1,D);
W_MAP = 0.001 * ones(1,D);

alpha = 0.5;
beta = 0.5;
lr = 0.5;   % learning rate for gradient descent
max_iter = 300;
t = y_train';

%% TRAIN

if method == 'GD'

    % calculating w_MAP using its gradient
    for e=1:max_iter
        grad = (t-(1./(1 + exp(-(W_MAP*X_train')))))*X_train + (W_MAP - prior_mean)*(inv(prior_variance))';
        W_MAP = W_MAP + (lr * grad);
    end
    
%     S_n = inv(prior_variance);
%     for i=1:N
%         S_n = S_n + (sigmoid(W_MAP*X_train(i,:)')*(1-sigmoid(W_MAP*X_train(i,:)')) .* (X_train(i,:)' * X_train(i,:)));
%     end
%     hess = S_n;

      hess = X_train' * (X_train' .* sigmoid(W_MAP*X_train').*(1-sigmoid(W_MAP*X_train')))' + inv(prior_variance);

end

if method == 'NR'

    for e=1:max_iter
        grad = (t-sigmoid(W_MAP*X_train'))*X_train + (W_MAP - prior_mean)*(inv(prior_variance))';
        hess = X_train' * (X_train' .* sigmoid(W_MAP*X_train').*(1-sigmoid(W_MAP*X_train')))' + inv(prior_variance);
        W_MAP = W_MAP - (lr * grad * inv(hess)');
    end
    
end
%% TEST

probs = [];
for i=1:length(y_test)
    Mu = W_MAP * X_test(i,:)';
    Sig = X_test(i,:) * hess * X_test(i,:)';
    KFunc = (1 + 8 ./ sqrt(3.14 .* Sig));
    prob = sigmoid(KFunc * Mu');
    probs = [probs; prob];
end

end

