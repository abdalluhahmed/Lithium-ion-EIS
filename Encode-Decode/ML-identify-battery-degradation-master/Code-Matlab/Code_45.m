%Mulit-Temperature EIS-Capacity GPR model tested on 35°C (Fig. 3(a))

%Mulit-Temperature EIS-Capacity GPR model tested on 35°C (Fig. 3(a))

%% load relevant files
clear all;


% filenames = {'EIS_data.txt', 'Capacity_data.txt'};
% for kk = 1:numel(filenames)
%     load(filenames{kk});
% end

load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_V\Training\Inputs_All.mat';
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_V\Training\Output_All.mat';

EIS_data = Inputs_All';
Capacity_data= Output_All;

load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_V\Testing\Input_45C02.mat'
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_V\Testing\Output_45C02.mat'
Input_45C02=Input_45C02';

EIS_data_45=Input_45C02;
Capacity_data_45=Output_45C02;


%% Training set of the GRP model 
mean = mean(EIS_data,1); std = std(EIS_data,1);      %EIS_data is the raw experimental EIS data of the training cells cycled at 25, 35 and 45°C. EIS spectra are collected at state V.
X_train = zscore (EIS_data);                         %X_train is the input of the model after normalization
Y_train = Capacity_data;                             %Capacity_data is the corresponding capacity of the training cells, defined as Y_train, the output of the model.

%Mulit-Temperature EIS-Capacity GPR model
meanfunc = @meanZero; hyp.mean = [];                 %mean function is zero
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);    %the Gaussian likelihood
covfunc = @covSEiso; hyp.cov = [0; 0];               %Squared Exponential covariance function
hyp_EIS_Capacity = minimize(hyp, @gp, -10000, @infGaussLik, meanfunc, covfunc, likfunc, X_train, Y_train); %set hyperparameters by optimizing the (log) marginal likelihood





%% Testing set of the GPR model
X_test_45C02 = (EIS_data_45-mean)./std;           %EIS_data_35C02 is the EIS data of 35C02 cell.

%Capacity Estimation of the testing cell 
[Y_test_cap_45C02,Y_test_cap_35C02_var] = gp(hyp_EIS_Capacity,@infGaussLik,meanfunc,covfunc,likfunc,X_train, Y_train, X_test_45C02);  %Y_test_cap_35C02 is the estimated capacity. Y_test_cap_35C02_var is the uncertainty.




%%


%% The Plot of the estimated capacity (Fig. 3(a))
% Assuming cycleNumbers contains your actual cycle numbers
% Replace cycleNumbers with your actual x-axis data array
SOH_capacity45C02 = (Capacity_data_45 / Capacity_data_45(1,1)) * 100;
SOH_Y_test_cap_45C02 = (Y_test_cap_45C02 / Y_test_cap_45C02(1,1)) * 100;
% The Plot of the estimated capacity (Fig. 3(a))
% Define variability percentage for confidence interval
variability_percentage = 0.03; % Example variability
upper_bound = SOH_Y_test_cap_45C02 * (1 + variability_percentage);
lower_bound = SOH_Y_test_cap_45C02 * (1 - variability_percentage);


% Calculate and fill the confidence interval
% Calculate and fill the confidence interval
% Calculate and fill the confidence interval
% Define the cycle number vector
% Assuming SOH_Y_test_cap_45C02 is already calculated and is a column vector
% Calculate cycleNumbers based on the length of SOH_Y_test_cap_45C02
cycleNumbers = (2:2:(2*length(SOH_Y_test_cap_45C02)))'; 

% Now calculate the upper and lower bounds
variability_percentage = 0.03; % Adjust this as per your requirement
upper_bound = SOH_Y_test_cap_45C02 * (1 + variability_percentage);
lower_bound = SOH_Y_test_cap_45C02 * (1 - variability_percentage);

% Ensure that upper_bound and lower_bound are column vectors
upper_bound = upper_bound(:);
lower_bound = lower_bound(:);

figure;
f = [upper_bound; flipdim(lower_bound, 1)]; 
h = fill([[2:2:618]'; flipdim([2:2:618]', 1)], f, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
set(gcf, 'color', 'w');
hold on; 
p1 = plot([2:2:618], SOH_capacity45C02, 'b-*', 'LineWidth', 1, 'MarkerSize', 18);
p2 = plot([2:2:618], SOH_Y_test_cap_45C02, 'r-*', 'LineWidth', 1, 'MarkerSize', 18);
xlim([0 600]);
ylim([65 100]);

% Set X and Y labels with bold font and size 40
xlabel('Cycle number', 'FontWeight', 'bold', 'FontSize', 40);
ylabel('SOH (%)', 'FontWeight', 'bold', 'FontSize', 40);
title('45C02', 'FontWeight', 'bold', 'FontSize', 40);

% Modify legend to include only the plot markers, not the confidence interval
lgd = legend([p1, p2], 'Actual SOH (45C02)', 'Estimated SOH (45C02)');
set(lgd, 'FontSize', 40, 'FontWeight', 'bold');

% Set the font size and style of the tick labels on both axes
set(gca, 'FontSize', 40, 'FontWeight', 'bold');

% Turn the grid on
grid on;

% Turn the box on
box on;




%%

% Calculate the percentage capacities
SOH_capacity45C02 = (Capacity_data_45 / Capacity_data_45(1,1)) ; %%%%%%%%%%%%%%%%%%
SOH_Y_test_cap_45C02 = (Y_test_cap_45C02 / Y_test_cap_45C02(1,1)) ; %%%%%%%%%%%%%

clear mean
% Assuming you have loaded the 'capacity35C02.txt' data into the variable 'capacity35C02'

% Assuming you have loaded the 'capacity35C02.txt' data into the variable 'capacity35C02'

% Calculate Mean Squared Error (MSE) between 'capacity35C02' and 'Y_test_cap_35C02'
mean_error = mean(SOH_Y_test_cap_45C02 - SOH_capacity45C02);
mse = mean((SOH_capacity45C02 - SOH_Y_test_cap_45C02).^2);
rmse = sqrt(mse);

% Calculate R-squared (R^2) between 'capacity35C02' and 'Y_test_cap_35C02'
y_mean = mean(SOH_capacity45C02);
ss_total = sum((SOH_capacity45C02 - y_mean).^2);
ss_residual = sum((SOH_capacity45C02 - SOH_Y_test_cap_45C02).^2);
r_squared = 1 - (ss_residual / ss_total);

% Calculate Mean Absolute Percentage Error (MAPE) as a decimal
mape = mean(abs((SOH_Y_test_cap_45C02 - SOH_capacity45C02) ./ SOH_Y_test_cap_45C02));
% Calculate Mean Error

% Display the results
fprintf('Mean Error between capacity35C02 and Y_test_cap_35C02: %.4f\n', mean_error);
fprintf('Mean Squared Error (MSE) between capacity35C02 and Y_test_cap_35C02: %.4f\n', mse);
fprintf('Root Mean Squared Error (RMSE) between capacity35C02 and Y_test_cap_35C02: %.4f\n', rmse);
fprintf('R-squared (R^2) between capacity35C02 and Y_test_cap_35C02: %.4f\n', r_squared);
fprintf('Mean Absolute Percentage Error (MAPE) between capacity35C02 and Y_test_cap_35C02: %.4f\n', mape);




