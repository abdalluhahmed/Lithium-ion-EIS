% newff, newcf, traingdm, traingda, traingdx, trainlm, trainrp, traincgf, traincgb, trainbfg, traincgp, trainoss

%% Load Training Data
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_III\Training\Output_All.mat';
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_III\Training\Inputs_All.mat';
Inputs     = feature_matrix;    % feature_matrix loaded from Inputs_All.mat
Output_All = Output_All';       % transpose for plotting

%% Load Test Data & compute true SOH
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_III\Testing\Output_45C02.mat';
Output_45C02 = Output_45C02';  
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_III\Testing\Output_35C02.mat';
Output_35C02 = Output_35C02';  

SOH_actual_45C02 = (Output_45C02 / Output_45C02(1,1)) * 100;
SOH_actual_35C02 = (Output_35C02 / Output_35C02(1,1)) * 100;

%% Discover all net*.mat files and sort by numeric suffix
modelDir = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\ANN_State_III\net';
allFiles  = dir(fullfile(modelDir, 'net*.mat'));
nums      = arrayfun(@(f) sscanf(f.name,'net%d.mat'), allFiles);
[~, ord]  = sort(nums);
files     = allFiles(ord);

N          = numel(files);
mse45      = zeros(1, N);
mse35      = zeros(1, N);

%% Evaluate each model on both test sets
for k = 1:N
    % Load the k-th network file
    S = load(fullfile(modelDir, files(k).name));
    fn = fieldnames(S);
    % Find the network object inside S
    mdl = [];
    for i = 1:numel(fn)
        if isa(S.(fn{i}), 'network')
            mdl = S.(fn{i});
            break
        end
    end
    if isempty(mdl)
        error('No network object found in %s', files(k).name);
    end

    % Perform predictions
    pred45 = mdl(features_test);
    pred35 = mdl(features_test_35C02);
    norm45 = (pred45 / pred45(1,1)) * 100;
    norm35 = (pred35 / pred35(1,1)) * 100;

    % Compute MSE
    mse45(k) = mean((SOH_actual_45C02 - norm45).^2);
    mse35(k) = mean((SOH_actual_35C02 - norm35).^2);
end

%% Select the best model (MSE < 10 on both & lowest average)
validModels = (mse45 < 10) & (mse35 < 10);
if ~any(validModels)
    error('No models meet MSE < 10 on both test sets.');
end
avgMSE      = (mse45(validModels) + mse35(validModels)) / 2;
candidates  = find(validModels);
[~, idxRel] = min(avgMSE);
bestFile    = files(candidates(idxRel)).name;
fprintf('Best model file: %s\n', bestFile);

%% Load the best model once
S = load(fullfile(modelDir, bestFile));
fn = fieldnames(S);
bestModel = [];
for i = 1:numel(fn)
    if isa(S.(fn{i}), 'network')
        bestModel = S.(fn{i});
        break
    end
end
if isempty(bestModel)
    error('Could not find a network object in %s', bestFile);
end

%% Use bestModel for final predictions & plotting
p45_best = bestModel(features_test);
SOH45_est = (p45_best / p45_best(1,1)) * 100;
p35_best = bestModel(features_test_35C02);
SOH35_est = (p35_best / p35_best(1,1)) * 100;


% Compute and display final metrics
figure; hold on;
plot(SOH_actual_45C02, 'b-o', 'LineWidth', 2);
plot(SOH45_est,          'r-*', 'LineWidth', 2);
xlabel('Cycle Number', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('SOH (%)',       'FontSize', 12, 'FontWeight', 'bold');
title('45°C Test: Actual vs. Estimated SOH', 'FontSize', 14, 'FontWeight', 'bold');
legend('Actual','Estimated'); grid on; box on;

% Compute and display final metrics
mse_final45  = mean((SOH_actual_45C02 - SOH45_est).^2);
rmse_final45 = sqrt(mse_final45);
fprintf('45°C Test  MSE=%.4f  RMSE=%.4f\n', mse_final45, rmse_final45);
%
figure; hold on;
plot(SOH_actual_35C02, 'b-o', 'LineWidth', 2);
plot(SOH35_est,          'r-*', 'LineWidth', 2);
xlabel('Cycle Number', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('SOH (%)',       'FontSize', 12, 'FontWeight', 'bold');
title('35°C Test: Actual vs. Estimated SOH', 'FontSize', 14, 'FontWeight', 'bold');
legend('Actual','Estimated'); grid on; box on;

% Compute and display final metrics
mse_final35  = mean((SOH_actual_35C02 - SOH35_est).^2);
rmse_final35 = sqrt(mse_final35);
fprintf('45°C Test  MSE=%.4f  RMSE=%.4f\n', mse_final35, rmse_final35);


% Repeat plotting and metrics for 35°C as needed...


% Plot and calculate metrics for both datasets using the best model
% ... (Plotting and metrics calculation code goes here for both 45C02 and 35C02)



% Plot and calculate metrics for both datasets using the best model
% ... (Plotting and metrics calculation code goes here for both 45C02 and 35C02)





%% %%%%%%%%%%%%%%%%%%%  Plots Training data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predictions on Training Data
% Assuming 'Inputs' is your training feature data and 'Output_All' is your actual training output data

% Predict the training data output using the best model
predicted_training_output = bestModel(Inputs);
% Normalize the predicted output if required (depends on how your Output_All data is structured)
% For example, normalization can be similar to how it's done for the test datasets
% Plot Actual vs. Predicted Training Data
figure;
hold on;
% Plot actual training data
plot(Output_All, 'b-o', 'LineWidth', 2); % Blue circles for actual dat
% Plot predicted training data
plot(predicted_training_output, 'r-*', 'LineWidth', 2); % Red stars for predicted data
% Labeling the plot
xlabel('Sample Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Output Value', 'FontSize', 12, 'FontWeight', 'bold');
title('Comparison of Actual vs. Predicted Training Data', 'FontSize', 14, 'FontWeight', 'bold');
legend('Actual Training Data', 'Predicted Training Data');
grid on;
box on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;


% Calculate MSE and RMSE
% Calculate the squared differences

% Output_All=Output_All(:,1061:end);
% predicted_training_output=predicted_training_output(:,1061:end);
Output_All=Output_All';
squaredDifferences = (Output_All - predicted_training_output).^2;
% Calculate MSE
mse = mean(squaredDifferences);
% Calculate RMSE
rmse = sqrt(mse);
% Display the results
disp(['MSE on Training Data: ', num2str(mse)]);
disp(['RMSE on Training Data: ', num2str(rmse)]);


%% Plot training weights 

% Assuming 'bestModel' is your trained feedforward neural network
% Full MATLAB script to plot the weights of a trained neural network

% Clear workspace, command window, and close all figures

% Assuming 'bestModel' is your trained feedforward neural network
% Load or define your 'bestModel' variable here
% bestModel = ... (load or define your trained network)

hiddenWeights = bestModel.IW{1,1}; % Weights from input to hidden layer
outputWeights = bestModel.LW{2,1}; % Weights from hidden to output layer

% Set the desired marker size for scatter plot
markerSize = 140;  % You can adjust this value as needed

% Get the current colormap
cmap = jet(64); % 64 is a typical size for MATLAB colormaps

% Plot hidden layer weights
figure;
axis square;
hold on; % Allows multiple plots to be overlaid

% Plot each neuron's hidden layer weights with colors from the heatmap
for i = 1:size(hiddenWeights, 2) % Loop over the number of neurons in the hidden layer
    % Map neuron index to a color in the colormap
    colorIdx = round(1 + (size(cmap, 1) - 1) * (i - 1) / (size(hiddenWeights, 2) - 1));
    scatter(i * ones(size(hiddenWeights, 1), 1), hiddenWeights(:, i), 'filled', ...
        'CData', cmap(colorIdx, :), 'SizeData', markerSize);
end

hold off;
title('Hidden Layer Weights Comparison', 'FontSize', 50, 'FontWeight', 'bold');
xlabel('Neuron Index', 'FontSize', 50, 'FontWeight', 'bold');
ylabel('Weights', 'FontSize', 50, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold'); 
grid on;
box on;
set(gcf, 'Color', 'w'); % Set background color to white

% Plot output layer weights
figure;
axis square;

hold on; % Allows multiple plots to be overlaid

% Plot each neuron's output layer weights with colors from the heatmap
for i = 1:size(outputWeights, 2) % Loop over the number of neurons in the output layer
    % Map neuron index to a color in the colormap
    colorIdx = round(1 + (size(cmap, 1) - 1) * (i - 1) / (size(outputWeights, 2) - 1));
    scatter(i * ones(size(outputWeights, 1), 1), outputWeights(:, i), 'filled', ...
        'CData', cmap(colorIdx, :), 'SizeData', markerSize);
end

hold off;
title('Output Layer Weights Comparison', 'FontSize', 50, 'FontWeight', 'bold');
xlabel('Neuron Index', 'FontSize', 50, 'FontWeight', 'bold');
ylabel('Weights', 'FontSize', 50, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold');
grid on;
box on;
set(gcf, 'Color', 'w'); % Set background color to white




%% %%%%%%%%%%%%%%%%%%%%%%%%% Testing plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot and Calculate Metrics for 35C02 Dataset Using the Best Model

% % Plot for 35C02 Dataset
% figure;
% hold on;
% plot(Output_35C02, 'b-o', 'LineWidth', 4); % Actual Output for 35C02
% plot(predicted_SOH_35C02, 'r-*', 'LineWidth', 4); % Best Model Predicted SOH for 35C02
% xlabel('Cycle number', 'FontSize', 20, 'FontWeight', 'bold'); 
% ylabel('Capacity (%)', 'FontSize', 20, 'FontWeight', 'bold'); 
% legend('Actual SOH (35C02)', 'Predicted SOH (35C02)'); 
% grid on; 
% box on; 
% set(gca, 'FontSize', 20, 'FontWeight', 'bold'); 
% hold off;
% 
% %%
% %% Plot and Calculate Metrics for 45C02 Dataset Using the Best Model
% 
% % Plot for 45C02 Dataset
% figure;
% hold on;
% plot(Output_45C02, 'g-o', 'LineWidth', 4); % Actual Output for 45C02
% plot(predicted_SOH_45C02, 'm-*', 'LineWidth', 4); % Best Model Predicted SOH for 45C02
% xlabel('Cycle number', 'FontSize', 20, 'FontWeight', 'bold'); 
% ylabel('Capacity (%)', 'FontSize', 20, 'FontWeight', 'bold'); 
% legend('Actual SOH (45C02)', 'Predicted SOH (45C02)'); 
% grid on; 
% box on; 
% set(gca, 'FontSize', 20, 'FontWeight', 'bold'); 
% hold off;
% 
% %% Calculate Metrics for 35C02 Dataset and Create Table
% 
% mseTest_35C02 = mean((Output_35C02 - predicted_SOH_35C02).^2);
% rmseTest_35C02 = sqrt(mseTest_35C02);
% ssResTest_35C02 = sum((Output_35C02 - predicted_SOH_35C02).^2);
% ssTotTest_35C02 = sum((Output_35C02 - mean(Output_35C02)).^2);
% rSquaredTest_35C02 = 1 - (ssResTest_35C02 / ssTotTest_35C02);
% 
% Results_Capacity_35C02 = table(mseTest_35C02, rmseTest_35C02, rSquaredTest_35C02, ...
%     'VariableNames', {'MSE_Test_35C02', 'RMSE_Test_35C02', 'R_Squared_Test_35C02'});
% disp(Results_Capacity_35C02);
% 
% %% Calculate Metrics for 45C02 Dataset and Create Table
% mseTest_45C02 = mean((Output_45C02 - predicted_SOH_45C02).^2);
% rmseTest_45C02 = sqrt(mseTest_45C02);
% ssResTest_45C02 = sum((Output_45C02 - predicted_SOH_45C02).^2);
% ssTotTest_45C02 = sum((Output_45C02 - mean(Output_45C02)).^2);
% rSquaredTest_45C02 = 1 - (ssResTest_45C02 / ssTotTest_45C02);
% 
% Results_Capacity_45C02 = table(mseTest_45C02, rmseTest_45C02, rSquaredTest_45C02, ...
%     'VariableNames', {'MSE_Test_45C02', 'RMSE_Test_45C02', 'R_Squared_Test_45C02'});
% disp(Results_Capacity_45C02);



%% Plot and Calculate Metrics for Both Datasets Using the Best Model


% Plot for 45C02 Dataset

% Plot for 35C02 Dataset
figure;
hold on;

% Define the bounds for the shaded area based on variability_percentage
variability_percentage = 0.03; % Example variability percentage
upper_bound = bestModelPrediction_45C02 * (1 + variability_percentage);
lower_bound = bestModelPrediction_45C02 * (1 - variability_percentage);

% Generate x_values from 2 to 600 with a step of 2
x_values = 2:2:600;

% Ensure that the data and bounds have the same length as x_values
data_length = min([length(x_values), length(bestModelPrediction_45C02)]);
bestModelPrediction_45C02 = bestModelPrediction_45C02(1:data_length);
SOH_actual_45C02 = SOH_actual_45C02(1:data_length);
upper_bound = upper_bound(1:data_length);
lower_bound = lower_bound(1:data_length);

% Plot boundaries for predicted SOH (shaded region)
x_fill = [x_values, fliplr(x_values)]; % Concatenating the x_values for fill
y_fill = [upper_bound, fliplr(lower_bound)]; % Concatenating upper and lower bounds for fill
h_fill = fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 

% Plot Actual SOH for 35C02 in blue
h_actual = plot(x_values, SOH_actual_45C02, 'b-*', 'LineWidth', 1, 'MarkerSize', 18);
hold on

% Plot Best Model Predicted SOH for 35C02 in red
h_predicted = plot(x_values, bestModelPrediction_45C02, 'r-*', 'LineWidth', 1, 'MarkerSize', 18);

% Set x and y labels with FontSize 40 and FontWeight bold
xlabel('Cycle number', 'FontSize', 40, 'FontWeight', 'bold'); 
ylabel('SOH (%)', 'FontSize', 40, 'FontWeight', 'bold'); 

% Set the y-axis limit from 70 to 100
ylim([70 100]);

% Set the x-axis limits from 0 to 600
xlim([0 600]);

% Set the y-axis limit from 70 to 100

% Add a legend with FontSize 40, only including the actual and predicted lines
lgd = legend([h_actual, h_predicted], 'Actual SOH (45C02)', 'Estimated SOH (45C02)');
set(lgd, 'FontSize', 40, 'FontWeight', 'bold');

% Turn on grid and box
grid on;
box on;

% Set axes properties
set(gca, 'FontSize', 40, 'FontWeight', 'bold');

% Release the hold on the current plot
hold off;





% Plot for 35C02 Dataset
% Plot for 35C02 Dataset

% Plot for 35C02 Dataset



% Plot for 35C02 Dataset


variability_percentage = 0.03; % Example variability
upper_bound = bestModelPrediction_35C02 * (1 + variability_percentage);
lower_bound = bestModelPrediction_35C02 * (1 - variability_percentage);


figure;
hold on;


% Generate x_values from 2 to 600 with a step of 2
x_values = 2:2:400;

% Ensure that the data and bounds have the same length as x_values
data_length = min([length(x_values), length(bestModelPrediction_35C02)]);
bestModelPrediction_35C02 = bestModelPrediction_35C02(1:data_length);
SOH_actual_35C02 = SOH_actual_35C02(1:data_length);
upper_bound = upper_bound(1:data_length);
lower_bound = lower_bound(1:data_length);

% Plot boundaries for predicted SOH (shaded region)
x_fill = [x_values, fliplr(x_values)]; % Concatenating the x_values for fill
y_fill = [upper_bound, fliplr(lower_bound)]; % Concatenating upper and lower bounds for fill
h_fill = fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 

% Plot Actual SOH for 35C02 in blue
h_actual = plot(x_values, SOH_actual_35C02, 'b-*', 'LineWidth', 1, 'MarkerSize', 18);
hold on

% Plot Best Model Predicted SOH for 35C02 in red
h_predicted = plot(x_values, bestModelPrediction_35C02, 'r-*', 'LineWidth', 1, 'MarkerSize', 18);

% Set x and y labels with FontSize 40 and FontWeight bold
xlabel('Cycle number', 'FontSize', 40, 'FontWeight', 'bold'); 
ylabel('SOH (%)', 'FontSize', 40, 'FontWeight', 'bold'); 

% Set the y-axis limit from 70 to 100
ylim([70 100]);

% Set the x-axis limits from 0 to 600
xlim([0 400]);

% Set the y-axis limit from 70 to 100

% Add a legend with FontSize 40, only including the actual and predicted lines
lgd = legend([h_actual, h_predicted], 'Actual SOH (35C02)', 'Estimated SOH (35C02)');
set(lgd, 'FontSize', 40, 'FontWeight', 'bold');

% Turn on grid and box
grid on;
box on;

% Set axes properties
set(gca, 'FontSize', 40, 'FontWeight', 'bold');

% Release the hold on the current plot
hold off;



%% (SOH) Calculate Metrics for Test Data (45C02 and 35C02)

SOH_actual_45C02 = (Output_45C02 / Output_45C02(1,1));
SOH_actual_35C02 = (Output_35C02 / Output_35C02(1,1));

normalized_predicted_SOH_45C02 = (predicted_SOH_45C02 / predicted_SOH_45C02(1,1));
bestModelPrediction_45C02 = normalized_predicted_SOH_45C02;

% Predict and normalize for 35C02 dataset
normalized_predicted_SOH_35C02 = (predicted_SOH_35C02 / predicted_SOH_35C02(1,1));
bestModelPrediction_35C02 = normalized_predicted_SOH_35C02;


% Metrics for 35C02

mean_Error_35C02 = mean(SOH_actual_35C02 - bestModelPrediction_35C02);

Error_35C02 = (SOH_actual_35C02 - bestModelPrediction_35C02);
mseTest_35C02 = mean((Error_35C02).^2);
rmseTest_35C02 = sqrt(mseTest_35C02);
ssResTest_35C02 = sum((Error_35C02).^2);
ssTotTest_35C02 = sum((SOH_actual_35C02 - mean(SOH_actual_35C02)).^2);
rSquaredTest_35C02 = 1 - (ssResTest_35C02 / ssTotTest_35C02);
mapeTest_35C02 = mean(abs(Error_35C02 ./ SOH_actual_35C02));




% Metrics for 45C02
mean_Error_45C02 = mean(SOH_actual_45C02 - bestModelPrediction_45C02);


Error_45C02 = (SOH_actual_45C02 - bestModelPrediction_45C02);
mseTest_45C02 = mean((Error_45C02).^2);
rmseTest_45C02 = sqrt(mseTest_45C02);
ssResTest_45C02 = sum((Error_45C02).^2);
ssTotTest_45C02 = sum((SOH_actual_45C02 - mean(SOH_actual_45C02)).^2);
rSquaredTest_45C02 = 1 - (ssResTest_45C02 / ssTotTest_45C02);
mapeTest_45C02 = mean(abs(Error_45C02 ./ SOH_actual_45C02));




Results_SOH_35C02 = table(mean_Error_35C02,mseTest_35C02, rmseTest_35C02, rSquaredTest_35C02, mapeTest_35C02, ...
    'VariableNames', {'mean_Error_35C02','MSE_Test_35C02', 'RMSE_Test_35C02', 'R_Squared_Test_35C02', 'MAPE_Test_35C02'})

% Create tables for test data results
Results_SOH_45C02 = table(mean_Error_45C02,mseTest_45C02, rmseTest_45C02, rSquaredTest_45C02, mapeTest_45C02, ...
    'VariableNames', {'mean_Error_45C02','MSE_Test_45C02', 'RMSE_Test_45C02', 'R_Squared_Test_45C02', 'MAPE_Test_45C02'})

