function [res] = recall(preds,labels)
%RECALL measure

K = max(labels);    %number of classes

recalls = [];
for k=1:K
    recalls = [recalls; length(find(labels==k & labels==preds))/length(find(labels==k))];
end

recalls(isnan(recalls))=1;  
recalls(isinf(recalls))=0;  

res = sum(recalls)/K;

end

