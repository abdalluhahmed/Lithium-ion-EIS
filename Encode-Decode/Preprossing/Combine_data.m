

path = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets-Extracted\Training';

% List of file names
file_names = {'EIS_state_I_25C01', 'EIS_state_I_25C02', 'EIS_state_I_25C03', 'EIS_state_I_25C04', 'EIS_state_I_35C01', 'EIS_state_I_45C01'};

% Initialize a cell array to hold the combined data
Inputs_All = {};

% Initialize a cell array to hold the keys for each group
keys = {};

% Loop through each file name
for i = 1:length(file_names)
    % Create the full file path
    file_path = fullfile(path, [file_names{i}, '.mat']);
    
    % Load the file
    file = load(file_path);
    
    % Get the field name of the loaded data
    fieldName = fieldnames(file);
    
    % Extract the group from the file name
    token = regexp(file_names{i}, 'state_(\w+)_', 'tokens', 'once');
    key = token{1};
    
    % Find the index of the group that this file belongs to
    idx = find(strcmp(keys, key));
    
    % If the group does not exist yet, create a new group
    if isempty(idx)
        keys{end+1} = key;
        idx = length(keys);
        Inputs_All{idx} = file.(fieldName{1});
    % If the group already exists, concatenate the data
    else
        Inputs_All{idx} = [Inputs_All{idx}, file.(fieldName{1})];
    end
end

% Define the desired order of the keys
desired_order = {'I', 'II', 'III', 'IV', 'IX', 'V', 'VI'};

% Sort the keys in the desired order
[~, idx] = ismember(desired_order, keys);
sorted_Inputs_All = Inputs_All(idx);

% Concatenate the sorted groups into a single matrix
Inputs_All_Matrix = cat(2, sorted_Inputs_All{:});
