clc; clear;

%% =========================================================================
%  PARALLEL POOL (USE 6 CORES)
% =========================================================================
delete(gcp('nocreate'));
parpool(6);

%% =========================================================================
%  FILE PATHS
% =========================================================================
folder = "D:\CLASSES\FALL 2025\CE 7700 COMPUTATIONAL METHODS\PROJECT\TimeSeriesADCIRC\Processed";

file_noaa  = fullfile(folder, "noaa_8729108_Panama_City_MSL_19900101_to_20201231_TWL_TIDE_SURGE_merged.csv");
file_adcirc = fullfile(folder, "Panama_City_Station_303_Tide_NTR.csv");

fprintf("Loading NOAA and ADCIRC datasets...\n");

%% =========================================================================
%  LOAD NOAA
% =========================================================================
NOAA = readtable(file_noaa);
NOAA.time = datetime(NOAA.time);
NOAA = renamevars(NOAA, {'TWL','tide'}, {'TWL_noaa','tide_noaa'});

%% =========================================================================
%  LOAD ADCIRC
% =========================================================================
ADCIRC = readtable(file_adcirc);
ADCIRC.time = datetime(ADCIRC.Datetime);
ADCIRC = ADCIRC(:, {'time','NTR'});
ADCIRC.Properties.VariableNames = {'time','ntr_adcirc'};

%% =========================================================================
%  SYNCHRONIZE DATASETS
% =========================================================================
D = innerjoin(NOAA, ADCIRC, "Keys","time");
fprintf("Merged dataset rows = %d\n", height(D));

%% =========================================================================
%  BUILD FEATURES + NONLINEAR RESIDUAL
% =========================================================================
D.linearTWL = D.tide_noaa + D.ntr_adcirc;
D.r = D.TWL_noaa - D.linearTWL;

%% =========================================================================
%  TRAIN/TEST SPLIT
% =========================================================================
train = D(D.time < datetime(2016,1,1), :);
test  = D(D.time >= datetime(2016,1,1), :);

features = {'tide_noaa','ntr_adcirc','Tide_Lag1','NTR_Lag1','Tide_Lag3','NTR_Lag3'};

%% Add lags
train.Tide_Lag1 = [NaN; train.tide_noaa(1:end-1)];
train.NTR_Lag1  = [NaN; train.ntr_adcirc(1:end-1)];

train.Tide_Lag3 = [NaN(3,1); train.tide_noaa(1:end-3)];
train.NTR_Lag3  = [NaN(3,1); train.ntr_adcirc(1:end-3)];

test.Tide_Lag1 = [NaN; test.tide_noaa(1:end-1)];
test.NTR_Lag1  = [NaN; test.ntr_adcirc(1:end-1)];

test.Tide_Lag3 = [NaN(3,1); test.tide_noaa(1:end-3)];
test.NTR_Lag3  = [NaN(3,1); test.ntr_adcirc(1:end-3)];

train = rmmissing(train);
test = rmmissing(test);

%% Prepare matrices
Xtrain = double(train{:,features});
Ytrain = double(train.r);

Xtest = double(test{:,features});
Ytest = double(test.r);

%% =========================================================================
%  BASELINE 1 — LINEAR MODEL
% =========================================================================
TWL_pred_linear = test.tide_noaa + test.ntr_adcirc;

rmse_linear = sqrt(mean((test.TWL_noaa - TWL_pred_linear).^2));
corr_linear = corr(test.TWL_noaa, TWL_pred_linear);

%% =========================================================================
%  BASELINE 2 — BAGGED TREE ENSEMBLE (PARALLEL ENABLED)
% =========================================================================
fprintf("\nTraining Bagged Tree Model on 6 cores...\n");

% Tree template
t = templateTree('Reproducible',true);

% Parallel options
opts = statset('UseParallel',true);

model_tree = fitrensemble(Xtrain, Ytrain, ...
    'Method','Bag', ...
    'NumLearningCycles',200, ...
    'Learners', t, ...
    'Options', opts);

r_pred_tree = predict(model_tree, Xtest);
TWL_pred_tree = test.tide_noaa + test.ntr_adcirc + r_pred_tree;

rmse_tree = sqrt(mean((test.TWL_noaa - TWL_pred_tree).^2));
corr_tree = corr(test.TWL_noaa, TWL_pred_tree);

%% =========================================================================
%  MODEL 3 — BiLSTM (48 hour window)
% =========================================================================
window = 72;

fprintf("\nBuilding LSTM sequences using parfor...\n");

% ---------------- Build TRAIN sequences ----------------
Ntrain = height(train) - window;
Xtrain_seq = cell(Ntrain,1);
Ytrain_LSTM = zeros(Ntrain,1);

parfor i = 1:Ntrain
    Xtrain_seq{i} = [
        train.tide_noaa(i:i+window-1), ...
        train.ntr_adcirc(i:i+window-1)
    ]';
    Ytrain_LSTM(i) = train.r(i+window);
end

% ---------------- Build TEST sequences ----------------
Ntest = height(test) - window;
Xtest_seq = cell(Ntest,1);
Ytest_vec = zeros(Ntest,1);

parfor i = 1:Ntest
    Xtest_seq{i} = [
        test.tide_noaa(i:i+window-1), ...
        test.ntr_adcirc(i:i+window-1)
    ]';
    Ytest_vec(i) = test.r(i+window);
end

fprintf("TRAIN seq = %d, TEST seq = %d\n", Ntrain, Ntest);

%% =========================================================================
%  LSTM ARCHITECTURE
% =========================================================================
layers = [
    sequenceInputLayer(2)
    bilstmLayer(120,"OutputMode","last")
    dropoutLayer(0.2)
    fullyConnectedLayer(60)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions("adam", ...
    "MaxEpochs", 15, ...
    "MiniBatchSize", 256, ...
    "InitialLearnRate", 0.003, ...
    "Shuffle","every-epoch", ...
    "Verbose", false, ...
    "Plots","training-progress", ...
    "ExecutionEnvironment","parallel");

%% =========================================================================
%  TRAIN LSTM (PARALLEL)
% =========================================================================
fprintf("\nTraining LSTM on 6 cores...\n");
netLSTM = trainNetwork(Xtrain_seq, Ytrain_LSTM, layers, options);

%% =========================================================================
%  LSTM PREDICTION (parallel parfor)
% =========================================================================
fprintf("Predicting with LSTM on 6 cores...\n");
r_pred_LSTM = zeros(Ntest,1);

parfor i = 1:Ntest
    r_pred_LSTM(i) = predict(netLSTM, Xtest_seq{i});
end

% Align
aligned_obs  = test.TWL_noaa(window+1 : window + Ntest);
aligned_tide = test.tide_noaa(window+1 : window + Ntest);
aligned_ntr  = test.ntr_adcirc(window+1 : window + Ntest);

TWL_pred_LSTM = aligned_tide + aligned_ntr + r_pred_LSTM;

%% =========================================================================
%  LSTM PERFORMANCE
% =========================================================================
rmse_LSTM = sqrt(mean((aligned_obs - TWL_pred_LSTM).^2));
corr_LSTM = corr(aligned_obs, TWL_pred_LSTM);
fprintf("linear RMSE:  %.4f m\n", rmse_linear);
fprintf("linear Corr:   %.4f\n", corr_linear);
fprintf("Bagged Tree RMSE:  %.4f m\n", rmse_tree);
fprintf("Bagged Tree:   %.4f\n", corr_tree);
fprintf("\n========== LSTM PERFORMANCE (2016–2020) ==========\n");
fprintf("LSTM RMSE:  %.4f m\n", rmse_LSTM);
fprintf("LSTM Corr:   %.4f\n", corr_LSTM);

%% =========================================================================
%  ALIGN BAGGED TREE FOR SAME WINDOW
% =========================================================================
aligned_tree = TWL_pred_tree(window+1 : window + Ntest);
aligned_linear = aligned_tide + aligned_ntr;

%% =========================================================================
%  PLOT — TIME SERIES
%% =========================================================================
%  PLOT — TIME SERIES (TIME ON X AXIS)
% =========================================================================
aligned_time = test.time(window+1 : window + Ntest);

figure; hold on;

plot(aligned_time, aligned_obs, 'k','LineWidth',0.7);
plot(aligned_time, aligned_linear, 'r');
plot(aligned_time, aligned_tree, 'b');
plot(aligned_time, TWL_pred_LSTM, 'g');

legend("Observed","Linear","Bagged Tree","LSTM","Location","best");
title("TWL Comparison: Observed vs Linear vs Bagged vs LSTM");
xlabel("Time");
ylabel("Water Level (m)");
grid on;
datetick('x','keeplimits');   % optional: formats time ticks

figure; hold on;

plot(aligned_time, aligned_obs, 'k','LineWidth',0.7);
plot(aligned_time, aligned_linear, 'r','LineWidth',0.6);
%plot(aligned_time, aligned_tree, 'b');
plot(aligned_time, TWL_pred_LSTM, 'g');
legend("Observed","Linear","LSTM","Location","best");
%legend("Observed","Linear","Bagged Tree","LSTM","Location","best");
title("TWL Comparison: Observed vs Linear vs LSTM");
xlabel("Time");
ylabel("Water Level (m)");
grid on;
datetick('x','keeplimits');  
%% =========================================================================
%  PLOT — SCATTER
% =========================================================================
figure; hold on;

scatter(aligned_obs, aligned_linear, 5, 'r','filled');
scatter(aligned_obs, aligned_tree, 5, 'b','filled');
scatter(aligned_obs, TWL_pred_LSTM, 5, 'g','filled');

minVal = min(aligned_obs);
maxVal = max(aligned_obs);
plot([minVal maxVal], [minVal maxVal], 'k--','LineWidth',1.2);

legend("Linear","Bagged","LSTM","1:1 Line");
xlabel("Observed TWL");
ylabel("Predicted TWL");
title("Observed vs Predicted TWL (Linear, Bagged, LSTM)");
grid on;

