function [preds] = generativeClassification(X_train, X_test, y_train, y_test, K)
%GENERATIVECLASSIFICATIONWITHNAIVEBAYES 
%Reference: Bishop, C. M. (2006). Pattern recognition and machine learning - Springer (Chapter 4.2.1)
            %Andrew NG - Machine learning course lecture notes part IV
    
[N,D] = size(X_train);

%% TRAIN

% calculating prior probabilities and means
fprintf("calculating prior probabilities and means\n")
priors = [];
Mus = [];
for k=1:K
    priors = [priors; length(y_train(y_train == k))/N];
    Mus = [Mus; sum(X_train(find(y_train==k),:))/length(y_train(y_train == k))];
end

% calculating covariance matrix
% CovMat = zeros(D);
% for n=1:N
%   tempCov = (X_train(n,:) - Mus(y_train(n)))' * (X_train(n,:) - Mus(y_train(n)));
%    CovMat = CovMat + tempCov;
% end
% CovMat = CovMat./N;

fprintf("calculating covariance matrix\n")
covMat = cov(X_train);
% covMat = 1 * eye(D);
covMatInv = pinv(covMat);

% calculating each class coef parameters
fprintf("calculating each class coef parameters\n")

W = (covMatInv * Mus')';

W_0 = [];
for k=1:K
%     W = [W: covMatInv*
    W_0 = [W_0; -1/2*(Mus(k,:) * covMatInv * Mus(k,:)') + log(priors(k))];
end


%% TEST
fprintf("Testing ...")
res = W * X_test' + W_0;
% preds = softmax(res');
[~, preds] = max(res,[],1);
preds = preds';
end

