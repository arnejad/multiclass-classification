%% Initialization and reading dataset

% dividing randomly to test and train
test_P = 0.20; %97
[m,n] = size(fts) ;
idx = randperm(m);

K = max(labels);   % determining number of classes [1-K]

%% read files to fts and labels variables (or just click on your dataset in matlab files)

%% ML running

FPRs=[];
recalls = [];
precisions = [];
f1s = [];

% 5-Fold
for k=1:5
    % test data extraction
    fprintf("data number %d to %d as test\n", round((k-1)*test_P*m)+1, round(k*test_P*m))
    X_test = fts(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:); 
    y_test = labels(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:); 
    
    % train data extraction
    X_train = fts; y_train = labels;
    X_train(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:) = [];
    y_train(idx(round((k-1)*test_P*m)+1:round(k*test_P*m)),:) = [];
    

    % Run algorithm
    fprintf("Multinomial Log Reg:\n")
    preds = multinomialLogisticRegression(X_train, X_test, y_train, y_test, K); % change this func to other ML algorithms
    
    % evaluate
    rec = recall(preds, y_test); prec = precision(preds, y_test); f1Meas  = f1(prec,rec); fpr = FPR(preds, y_test);
    FPRs=[FPRs fpr]; recalls = [recalls rec]; precisions = [precisions prec]; f1s = [f1s f1Meas];
    fprintf("Precision: %d recall: %d f1: %d FPR: %d\n", prec, rec, f1Meas, fpr);

% 
end
fprintf("MEAN: Precision: %d recall: %d f1: %d\n", mean(precisions), mean(recalls), mean(f1s));
