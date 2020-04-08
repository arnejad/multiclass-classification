function [res] = precision(preds,labels)
%PRECISION measure

K = max(labels);    %number of classes

precs = [];
for k=1:K
    precs = [precs; length(find(labels == preds & preds==k))/length(find(preds==k))];
end

precs(isnan(precs))=1;  % one for not prediciting any of class that was not involved
precs(isinf(precs))=0;  % zero for class that was predicted non although existed

res = sum(precs)/K;

end

