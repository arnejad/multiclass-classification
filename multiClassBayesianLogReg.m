function [preds] = multiClassBayesianLogReg(X_train, X_test, y_train, y_test, K)
%MULTICLASSBAYESIANLOGREG uses binary Bayesian Logistic Regression and 
%one-vs-all(rest) strategy to solve a multi class classification
fprintf('Running Bayesian Logistic Regression\n')
probs = [];
for k=1:K
    fprintf('Class %d vs rest\n', k)
    train_lbls = y_train;
    % filtering labels
    train_lbls(train_lbls ~= k) = 0; 
    train_lbls(train_lbls == k) = 1; 
    
    test_lbls = y_test;
    test_lbls(test_lbls ~= k) = 0; 
    test_lbls(test_lbls == k) = 1; 
    
    res_probs = binaryBayesianLogReg(X_train, X_test, train_lbls, test_lbls);
    probs = [probs res_probs];
end

[~, preds] = max(probs, [], 2);

end

