function [res] = FPR(preds, labels)
%FPR calculates False Positive Rate measure

K = max(labels);    %number of classes

classSpecific = [];
for k=1:K
    classSpecific = [classSpecific; length(find(preds==k & labels~=preds))/length(find(labels~=k))];
end

classSpecific(isnan(classSpecific))=1;  % one for not prediciting any of class that was not involved
classSpecific(isinf(classSpecific))=0;  % zero for class that was predicted non although existed

res = sum(classSpecific)/K;

end

