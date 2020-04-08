function [preds] = gaussianNaiveBayesClassifier(X_train, X_test, y_train, y_test, K)
%GAUSSIANNAIVEBAYESCLASSIFIER 

%% INITIALIZATIONS
[N,D] = size(X_train);

Mus = zeros(K,D);
varSqs = zeros(K,D);


%% TRAIN

% Calcultaing priors
fprintf("calculating prior probabilities\n");

priors = [];
for k=1:K
    priors = [priors; length(y_train(y_train == k))/N];
end

% calculating means and covs
fprintf("calculating means and covs\n");

for k=1:K
    tmp = X_train(find(y_train==k),:);  % all the data of class k
   for d=1:D 
       Mus(k,d) = sum(tmp(:,d))/length(y_train(y_train == k));  %mean(tmp(:,d))
       varSqs(k,d) = (sum((tmp(:,d)-Mus(k,d)).^2) / length(y_train(y_train == k))); %var(tmp(:,d))
   end
end

Vars = varSqs.^(1/2);

%% TEST
%calculating test probabilities
fprintf("calculating test probabilities\n");
probs = [];
for i=1:length(y_test)
    samp_probs = [];
    for k=1:K
        classPred = priors(k);
        for d=1:D
            classPred = classPred * normpdf(X_test(i,d),Mus(k,d),Vars(k,d));
        end
        samp_probs = [samp_probs classPred];
    end
    probs = [probs; samp_probs];
end
[~,preds] = max(probs,[],2);
end

