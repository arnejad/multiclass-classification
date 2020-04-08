function [preds] = weightedVoting(X_train, X_test, y_train, y_test, K)
%WEIGHTEDVOTING is a proposed ensemble method

heus = [];

LRpreds = multinomialLogisticRegression(X_train,X_test, y_train, y_test,K);
rec = recall(LRpreds, y_test); prec = precision(LRpreds, y_test); 
heus = [heus; 2*f1(prec,rec)-FPR(LRpreds, y_test)];

BAYESpreds = multiClassBayesianLogReg(X_train,X_test, y_train, y_test,K);
rec = recall(BAYESpreds, y_test); prec = precision(BAYESpreds, y_test);
heus = [heus; 2*f1(prec,rec)-FPR(LRpreds, y_test)];

GENpreds = generativeClassification(X_train,X_test, y_train, y_test,K);
rec = recall(GENpreds, y_test); prec = precision(GENpreds, y_test);
heus = [heus; 2*f1(prec,rec)-FPR(LRpreds, y_test)];


SVMpreds = multiClassSVM(X_train,X_test, y_train, y_test,K);
rec = recall(SVMpreds, y_test); prec = precision(SVMpreds, y_test);
heus = [heus; 2*f1(prec,rec)-FPR(LRpreds, y_test)];

res = zeros(length(y_test),K);
for i=1:length(y_test)
    
    res(i,LRpreds(i)) = res(i,LRpreds(i)) + heus(1);
    res(i,BAYESpreds(i)) = res(i,BAYESpreds(i)) + heus(2);
    res(i,GENpreds(i)) = res(i,GENpreds(i)) + heus(3);
    res(i,SVMpreds(i)) = res(i,SVMpreds(i)) + heus(4);

end

[~, preds] = max(res,[],2);

end

