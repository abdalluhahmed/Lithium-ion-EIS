



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 25C01 ,  , and 35 , 45 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File paths
filepaths = {
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_25C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_II_25C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_III_25C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IV_25C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_25C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_V_25C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_VI_25C01.txt';


'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_35C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_II_35C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_III_35C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IV_35C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_35C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_V_35C01.txt';


'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_45C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_II_45C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_III_45C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IV_45C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_45C01.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_V_45C01.txt';

};


output_folder = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets-Extracted\Training';


% Loop over each file path
for i = 1:length(filepaths)
    % Read data from file
    filename = filepaths{i};
    data_EIS = dlmread(filename, '\t', 1, 0);

    % Determine the number of full groups of 60 in column 2
    num_groups = floor(size(data_EIS, 1) / 60);

    % Initialize the input matrix
    Input = zeros(120, num_groups);

    % Extract input matrix
    for j = 1:num_groups
        start_idx = (j - 1) * 60 + 1;
        end_idx = j * 60;

        % Extract data from columns 4 and 5 and flatten them into a 120-row vector
        col_vector = [data_EIS(start_idx:end_idx, 4); data_EIS(start_idx:end_idx, 5)];

        % Store this vector in the input matrix
        Input(:, j) = col_vector;
    end

    % Create variable name
    [~, name, ~] = fileparts(filename);
    varname = matlab.lang.makeValidName(name);
    
    % Save the input matrix as a .mat file in the output folder
    save(fullfile(output_folder, [varname '.mat']), 'Input');
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 25C02 , 25C03 , 25C04 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% File paths
filepaths = {

'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_25C02.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_II_25C02.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_III_25C02.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IV_25C02.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_25C02.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_V_25C02.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_VI_25C02.txt';

'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_25C03.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_II_25C03.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_III_25C03.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IV_25C03.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_25C03.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_V_25C03.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_VI_25C03.txt';

'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_25C04.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_II_25C04.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_III_25C04.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IV_25C04.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_IX_25C04.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_V_25C04.txt';
'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_VI_25C04.txt';



};

% Loop over each file path
for i = 1:length(filepaths)
    % Read data from file
    filename = filepaths{i};
    data_EIS = dlmread(filename, '\t', 1, 0);

    % Determine the number of full groups of 60 in column 2
    num_groups = floor(size(data_EIS, 1) / 60);

    % Initialize the input matrix
    Input = zeros(120, num_groups);

    % Extract input matrix
    for j = 1:num_groups
        start_idx = (j - 1) * 60 + 1;
        end_idx = j * 60;

        % Extract data from columns 4 and 5 and flatten them into a 120-row vector
        col_vector = [data_EIS(start_idx:end_idx, 4); data_EIS(start_idx:end_idx, 5)];

        % Store this vector in the input matrix
        Input(:, j) = col_vector;
    end

    % Create variable name
    [~, name, ~] = fileparts(filename);
    varname = matlab.lang.makeValidName(name);
    
    % Save the input matrix as a .mat file in the output folder
    save(fullfile(output_folder, [varname '.mat']), 'Input');
end
