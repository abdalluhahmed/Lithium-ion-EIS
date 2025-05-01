

%% Load output

load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets-Extracted\Training\Output_All.mat'

out_25C01 = Output_All(1:200,:);
out_25C02 = Output_All(201:450,:);
out_25C03 = Output_All(451:679,:);
out_25C04 = Output_All(680:760,:);
out_35C01 = Output_All(761:1059,:);
out_45C01 = Output_All(1060:end,:);

load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets-Extracted\Testing\Output_35C02.mat'
load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets-Extracted\Testing\Output_45C02.mat'
out_35C02 = Output_35C02;
out_45C02 = Output_45C02;

%% Training

% Define base file paths
eisBasePath = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_';
capacityBasePath = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\Capacity data\Data_Capacity_';

% List of dataset identifiers
datasets = {'25C01', '25C02', '25C03', '25C04', '35C01', '45C01'};

% Initialize final combined matrices
Inputs_All = [];
Outputs_All = [];

% Initialize a variable to store the previous maximum value
previousMax = Inf;

% Process each dataset
for k = 1:length(datasets)
    % Load EIS and capacity data
    eisFile = [eisBasePath, datasets{k}, '.txt'];
    eis_data = readmatrix(eisFile);
    capacityFile = [capacityBasePath, datasets{k}, '.txt'];
    capacity_data = readmatrix(capacityFile);

    % Adjust capacity data columns if more than 4
    if width(capacity_data) > 4
        capacity_data(:,[4, 5]) = [];
    end

    % Process data only for cycles without NaN in capacity data
    nonNaN_cycles = capacity_data(~isnan(capacity_data(:,4)), 2);
    unique_cycles = unique(intersect(nonNaN_cycles, eis_data(:,2)));
    Inputs = [];
    Outputs = [];

    % Flatten EIS data for each valid cycle and store in columns
    for i = 1:length(unique_cycles)
        cycle = unique_cycles(i);
        cycle_rows = eis_data(eis_data(:,2) == cycle, :);
        flattened_data = [cycle_rows(:, 4); cycle_rows(:, 5)];
        Inputs(:, i) = flattened_data;

        % Calculate the maximum value of column 4 for each valid cycle
        cycle_rows = capacity_data(capacity_data(:,2) == cycle, :);
        currentMax = max(cycle_rows(:,4));

        if currentMax > previousMax
            smallerValues = cycle_rows(cycle_rows(:,4) < previousMax, 4);
            if isempty(smallerValues)
                continue; % Skip this cycle
            else
                maxSmallerValue = max(smallerValues);
                Outputs(i) = maxSmallerValue;
            end
        else
            Outputs(i) = currentMax;
        end

        previousMax = Outputs(i);
    end

    % Assign inputs and outputs to individual variables
    eval(['Input_', datasets{k}, ' = Inputs;']);
    eval(['Output_', datasets{k}, ' = Outputs;']);
end

% Combine datasets
for k = 1:length(datasets)
    % Dynamic variable names for inputs, outputs, and reference output
    inputVar = ['Input_', datasets{k}];
    outputVar = ['Output_', datasets{k}];
    outVar = ['out_', datasets{k}];

    % Load the dataset-specific inputs, outputs, and reference output
    Inputs = eval(inputVar);
    Outputs = eval(outputVar);
    outData = eval(outVar);

    % Adjust the lengths of outData and Inputs to match the length of Outputs
    if size(Inputs, 2) < size(outData, 1)
        % Truncate outData to match Inputs' length
        outData = outData(1:size(Inputs, 2), :);
    elseif size(Inputs, 2) > size(outData, 1)
        % Extend outData with NaNs or zeros (as appropriate) to match Inputs' length
        outData = [outData; NaN(size(Inputs, 2) - size(outData, 1), size(outData, 2))];
    end

    % Append adjusted outData to Outputs_All and Inputs to Inputs_All
    Outputs_All = [Outputs_All; outData];
    Inputs_All = [Inputs_All, Inputs];
end

Inputs_All_Training = Inputs_All;
Outputs_All_Training = Outputs_All;

% Inputs_All and Outputs_All are now aligned and ready for further processing


% Inputs_All now contains merged and aligned input data
% Outputs_All now contains all the values from out_ matrices


% Inputs_All and Outputs_All now contain merged and aligned data

% Inputs_All and Outputs_All now contain merged and aligned data


%% Testing - Similar approach for test datasets

%% Testing - Similar approach for test datasets

%% Testing - Similar approach for test datasets

%% Testing - Similar approach for test datasets

% Additional datasets for testing
% Additional datasets for testing
testDatasets = {'35C02', '45C02'};


% Initialize matrices to store test inputs and outputs for '35C02' and '45C02'
testInputs_35C02 = [];
testOutputs_35C02 = [];
testInputs_45C02 = [];
testOutputs_45C02 = [];

% Process each test dataset
for k = 1:length(testDatasets)
    % Load EIS and capacity data
    eisFile = [eisBasePath, testDatasets{k}, '.txt'];
    eis_data = readmatrix(eisFile);
    capacityFile = [capacityBasePath, testDatasets{k}, '.txt'];
    capacity_data = readmatrix(capacityFile);

    % Adjust capacity data columns if more than 4
    if width(capacity_data) > 4
        capacity_data(:,[4, 5]) = [];
    end

    % Process data only for cycles without NaN in capacity data
    nonNaN_cycles = capacity_data(~isnan(capacity_data(:,4)), 2);
    unique_cycles = unique(intersect(nonNaN_cycles, eis_data(:,2)));
    
    % Initialize matrices to store inputs and outputs for this test dataset
    Inputs = [];
    Outputs = [];

    % Process data for each valid cycle and store in matrices
    for i = 1:length(unique_cycles)
        cycle = unique_cycles(i);
        cycle_rows_eis = eis_data(eis_data(:,2) == cycle, :);
        cycle_rows_capacity = capacity_data(capacity_data(:,2) == cycle, :);
        
        % Flatten EIS data for the cycle and store in a matrix
        flattened_data_eis = [cycle_rows_eis(:, 4); cycle_rows_eis(:, 5)];
        Inputs = [Inputs, flattened_data_eis];
        
        % Calculate the maximum value of column 4 for the cycle and store in a matrix
        max_value_capacity = max(cycle_rows_capacity(:, 4));
        Outputs = [Outputs; max_value_capacity];
    end

    % Store the inputs and outputs for '35C02' and '45C02' in separate matrices
    if strcmp(testDatasets{k}, '35C02')
        testInputs_35C02 = Inputs;
        testOutputs_35C02 = Outputs;
    elseif strcmp(testDatasets{k}, '45C02')
        testInputs_45C02 = Inputs;
        testOutputs_45C02 = Outputs;
    end
end

% Check and adjust dimensions for '35C02'
if size(testInputs_35C02, 2) > size(Output_35C02, 1)
    % Reduce the number of columns in testInputs_35C02
    testInputs_35C02 = testInputs_35C02(:, 1:size(Output_35C02, 1));
elseif size(testInputs_35C02, 2) < size(Output_35C02, 1)
    % Reduce the number of rows in Output_35C02
    Output_35C02 = Output_35C02(1:size(testInputs_35C02, 2), :);
end

% Check and adjust dimensions for '45C02'
if size(testInputs_45C02, 2) > size(Output_45C02, 1)
    % Reduce the number of columns in testInputs_45C02
    testInputs_45C02 = testInputs_45C02(:, 1:size(Output_45C02, 1));
elseif size(testInputs_45C02, 2) < size(Output_45C02, 1)
    % Reduce the number of rows in Output_45C02
    Output_45C02 = Output_45C02(1:size(testInputs_45C02, 2), :);
end


% testInputs_35C02 and testOutputs_35C02 contain data for '35C02'
% testInputs_45C02 and testOutputs_45C02 contain data for '45C02'

%%

% Find indices of NaN values in Outputs_All
nanIndices = isnan(Outputs_All_Training);
% Remove corresponding columns in Inputs_All and rows in Outputs_All
Inputs_All_Training(:, nanIndices) = [];
Outputs_All_Training(nanIndices) = [];
% Now, Inputs_All and Outputs_All are cleaned and aligned


Inputs_All=Inputs_All_Training;
Output_All=Outputs_All_Training;
Input_35C02=testInputs_35C02;
Input_45C02=testInputs_45C02;
Output_35C02=Output_35C02;
Output_45C02=Output_45C02;


% Clear all variables from the workspace except 'Output_45C02'
clearvars -except Inputs_All  Output_All Input_35C02 Input_45C02 Output_35C02 Output_45C02;





%% Save



% Define the path where you want to save the files
savePath = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_IX\Training';

% Check if the directory exists, if not, create it
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Save Inputs_All and Output_All in the specified directory
save(fullfile(savePath, 'Inputs_All.mat'), 'Inputs_All');
save(fullfile(savePath, 'Output_All.mat'), 'Output_All');




% Define the path where you want to save the files
savePath = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_IX\Testing';

% Check if the directory exists, if not, create it
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Save Inputs_All and Output_All in the specified directory
save(fullfile(savePath, 'Input_35C02.mat'), 'Input_35C02');
save(fullfile(savePath, 'Input_45C02.mat'), 'Input_45C02');
save(fullfile(savePath, 'Output_35C02.mat'), 'Output_35C02');
save(fullfile(savePath, 'Output_45C02.mat'), 'Output_45C02');



