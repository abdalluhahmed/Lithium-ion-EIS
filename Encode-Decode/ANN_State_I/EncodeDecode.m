
% 
%  clc
%  clear
%  close all



load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Training\Inputs_All'
Inputs_All=Inputs_All;

% Inputs_All(56:60,:)=[];
% Inputs_All(113:end,:)=[];
% 
% Inputs_All=Inputs_All;


% Input_35C02_75=Input_35C02(:,1:end);



% Append the statistics to the 'Inputs_All' matrix
Inputs_All = [Inputs_All];

% % Initialize the new row with zeros
% new_row = zeros(1, size(Inputs_All, 2));
% % Set values for different column ranges
% new_row(1:760) = 0.25;       % Columns 1 to 760
% new_row(761:1059) = 0.35;    % Columns 761 to 1059
% new_row(1060:end) = 0.45;    % Columns 1060 to the end
% % Append the new row to the Inputs_All matrix
% Inputs_All = [Inputs_All; new_row];


% % Define your scaling factor
% scaling_factor = 100;
% % Scale up the entire Inputs_All matrix
% Inputs_All = Inputs_All * scaling_factor;


% Assuming Inputs_All is your existing matrix
% Delete columns 680 to 760
% Inputs_All(:, 680:760) = [];


%%
% selectedRows = [1:30, 61:90];
% % New Inputs_All with 60 x 1358 dimensions
% Inputs_All = Inputs_All(selectedRows, :);

%%


% Split the matrix into A and B
% Split the matrix into A and B
% A = Inputs_All(1:60, :);  % First 60 rows
% B = Inputs_All(61:end, :);  % Second 60 rows
% 
% % Calculate the discrete derivative dA/dB
% dA_dB = zeros(size(A) - [1 0]);  % Preallocate for speed, result will be one less than the number of rows in A and B
% 
% % Loop through each column
% for col = 1:size(A, 2)
%     for row = 1:(size(A, 1) - 1)
%         dA_dB(row, col) = (A(row + 1, col) - A(row, col)) / (B(row + 1, col) - B(row, col));
%     end
% end


% dA_dB now contains the result of the division


%% Noise 
% 
% noise_std = 0.0001; % Standard deviation of the Gaussian noise
% 
% % Generate Gaussian noise with mean 0 and standard deviation noise_std
% noise = noise_std * randn(size(Inputs_All));
% 
% % Add noise to Inputs_All
% Inputs_All = Inputs_All + noise;

% Now Inputs_All_noisy contains the original data with added Gaussian noise


%%
% Assuming 'Inputs_All' is your 120x1358 matrix

% % Calculate statistics for each column
% min_values = min(Inputs_All, [], 1);         % Minimum values for each column
% max_values = max(Inputs_All, [], 1);         % Maximum values for each column
% mean_values = mean(Inputs_All, 1);           % Mean values for each column
% std_deviation = std(Inputs_All, 1);          % Standard deviation for each column
% median_values = median(Inputs_All, 1);       % Median values for each column
% range_values = range(Inputs_All, 1);         % Range values (max - min) for each column
% variance_values = var(Inputs_All, 1);        % Variance for each column
% sum_values = sum(Inputs_All, 1);             % Sum of values for each column
% kurtosis_values = kurtosis(Inputs_All, 1);   % Kurtosis for each column
% skewness_values = skewness(Inputs_All, 1);   % Skewness for each column
% 
% % Append the statistics as rows to 'Inputs_All'
% Inputs_All_with_stats = [Inputs_All; min_values; max_values; mean_values; std_deviation; ...
%                         median_values; range_values; variance_values; sum_values; ...
%                         kurtosis_values; skewness_values];
% 
% % Display the updated matrix size
% 
% Inputs_All=Inputs_All_with_stats;


%%

% % Original data matrix 'Inputs_All' is 120x1000
% % We will create a noisy version of this matrix
% 
% % Define the standard deviation of the Gaussian noise
% noise_std = 0; % for example, this can be changed depending on how much noise you want to add
% 
% % Generate Gaussian noise with zero mean and standard deviation 'noise_std'
% noise = noise_std * randn(size(Inputs_All));
% 
% % Add the noise to the original data to create a noisy version
% noisy_Inputs_All = Inputs_All + noise;
% 
% % Concatenate the original data with the noisy data
% % This results in a 240x1000 matrix
% Inputs_All = (noisy_Inputs_All);



%%
% Doubled_Inputs_All_Horizontal = [Inputs_All, Inputs_All];
% Inputs_All=Doubled_Inputs_All_Horizontal;

dQdV=Inputs_All;
orginal_data=dQdV;

 % Reshape the matrix to have each signal as a column
 orginal_data = reshape(dQdV, [], size(dQdV, 2));

% Set the size of the hidden layer in the autoencoder
hiddenSize = 20;

% Create and Train an Autoencoder
autoenc = trainAutoencoder(orginal_data, hiddenSize, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization', 0.00004, ...
    'SparsityRegularization', 0.004, ...
    'SparsityProportion', 0.15, ...
    'ScaleData', true, ...
    'UseGPU', true, ...
    'DecoderTransferFunction', 'purelin');

% Initialize a cell array to store the results
results = cell(1, size(dQdV, 2));

%%  (2) Loop over each individual signal
for i = 1:size(dQdV, 2)
    % Get the i-th signal
    single_signal = dQdV(:, i);
    
    % Test the model on the single signal
    single_features = encode(autoenc, single_signal);
    reconstructed_signal = decode(autoenc, single_features);
    
    % Store the results in the cell array
    results{i} = struct('original_signal', single_signal, 'encoded_features', single_features, 'reconstructed_signal', reconstructed_signal);
    
end

%% Feature matrix (3)
feature_matrix = zeros(hiddenSize, size(dQdV, 2));

%% Reconstructed matrix (4)
reconstructed_matrix = zeros(size(dQdV));

% Extract the encoded features for each signal and reconstruct the signals
for i = 1:size(dQdV, 2)
    result = results{i};
    encoded_features = result.encoded_features;
    feature_matrix(:, i) = encoded_features;
    
    reconstructed_signal = result.reconstructed_signal;
    reconstructed_matrix(:, i) = reconstructed_signal;
end



%% (5) Plot the original data, feature matrix, and reconstructed matrix in separate figures
% Define column ranges
column_ranges = {[1:261], [262:443], [444:664], [665:681], [682:1007], [1008:size(orginal_data, 2)]};

% Define titles for each subplot
subplot_titles = {'25C01', '25C02', '25C03', '25C04', '35C01', '45C01'};

% Font settings
axisLabelFontSize = 22;  % Font size for axis labels
titleFontSize = 22;  % Font size for titles
axisTicksFontSize = 20;  % Font size for axis tick values
fontWeight = 'bold';

% Segment size
segment_size = 60;

% Number of rows per figure
rows_per_figure = 3;

% Create two figures
for fig = 1:2
    figure('Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);

    % Determine the range of column indices for this figure
    col_range_indices = (fig-1)*rows_per_figure + (1:rows_per_figure);

    % Loop over the selected column ranges for this figure
    for idx = 1:length(col_range_indices)
        col_range_idx = col_range_indices(idx);

        % Generate colormap for the number of lines to be plotted
        cm = jet(length(column_ranges{col_range_idx}));

        % Loop to create 5 subplots for each column range
        for subplot_idx = 1:5
            % Calculate subplot position with adjusted y position
            subplot('Position', [
                (subplot_idx-1)*0.18+0.1,  % x position
                1 - idx*0.29 + 0.02,      % y position, adjusted to move up
                0.1,                      % width
                0.2                       % height
            ]);
            hold on;

            % Select and plot data based on subplot type
            if subplot_idx <= 2
                data_segment = (subplot_idx - 1) * segment_size + (1:segment_size);
                data = orginal_data(data_segment, column_ranges{col_range_idx});
            elseif subplot_idx == 3
                data = feature_matrix(:, column_ranges{col_range_idx});
            else
                data_segment = (subplot_idx - 4) * segment_size + (1:segment_size);
                data = reconstructed_matrix(data_segment, column_ranges{col_range_idx});
            end

            % Plot each line in the current column range
            for i = 1:length(column_ranges{col_range_idx})
                plot(data(:, i), 'LineWidth', 4, 'Color', cm(i, :));
            end

            hold off;
            grid on;
            box on;

            % Set y-axis labels
            if idx == ceil(rows_per_figure/2)
                switch subplot_idx
                    case 1
                        ylabel('Re(Z)/\Omega', 'FontWeight', fontWeight, 'FontSize', axisLabelFontSize);
                    case 2
                        ylabel('-Im(Z)/\Omega', 'FontWeight', fontWeight, 'FontSize', axisLabelFontSize);
                    case 3
                        ylabel('Features', 'FontWeight', fontWeight, 'FontSize', axisLabelFontSize);
                        xlabel('Feature Frequencies', 'FontWeight', fontWeight, 'FontSize', axisLabelFontSize); % X-axis label only for features
                    case {4, 5}
                        ylabel('Reconstructed', 'FontWeight', fontWeight, 'FontSize', axisLabelFontSize);
                end
            end

            % Set subplot title for the first subplot in each row
            if subplot_idx == 1
                title(subplot_titles{col_range_idx}, 'FontWeight', fontWeight, 'FontSize', titleFontSize);
            end

            set(gca, 'FontSize', axisTicksFontSize, 'FontWeight', fontWeight);
        end
    end
end




%% plot weights (6)
% Plot encoder weights
figure
for i = 1:5
    subplot(1, 5, i)
    encoder_weights = autoenc.EncoderWeights(:, i);
    bar(encoder_weights)
    title(['Encoder Weights - Neuron ', num2str(i)]);
    xlabel('Hidden Units');
    ylabel('Weight');
    set(gca, 'FontSize', 12, 'FontWeight', 'bold');
end

% Plot decoder weights
figure
for i = 1:5
    subplot(1, 5, i)
    decoder_weights = autoenc.DecoderWeights(:, i);
    bar(decoder_weights)
    title(['Decoder Weights - Neuron ', num2str(i)]);
    xlabel('Hidden Units');
    ylabel('Weight');
    set(gca, 'FontSize', 12, 'FontWeight', 'bold');
end


%% plot weights (7)


% Assuming autoenc is your trained autoencoder with EncoderWeights and DecoderWeights


% Assuming autoenc is your trained autoencoder with EncoderWeights and DecoderWeights
encoder_weights = autoenc.EncoderWeights;
decoder_weights = autoenc.DecoderWeights;

% Set the desired marker size for scatter plot
markerSize = 140;  % You can adjust this value as needed

% Create a new figure for the encoder scatter plot
figure;
axis square;
hold on; % Allows multiple plots to be overlaid

% Get the current colormap ('hot' for heatmap colors)
cmap = jet(64); % 64 is a typical size for MATLAB colormaps

% Plot each neuron's encoder weights with colors from the heatmap
for i = 1:20
    % Map neuron index to a color in the colormap
    colorIdx = round(1 + (size(cmap, 1) - 1) * (i - 1) / 19);
    scatter(i * ones(size(encoder_weights, 1), 1), encoder_weights(:, i), ...
        'filled', 'CData', cmap(colorIdx, :), 'SizeData', markerSize);
end

hold off;
title('Encoder Weights Comparison', 'FontSize', 50, 'FontWeight', 'bold');
xlabel('Neurons index', 'FontSize', 50, 'FontWeight', 'bold');
ylabel('Encoder weights', 'FontSize', 50, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold'); % Set axes tick label properties
grid on;
box on;
set(gcf, 'Color', 'w'); % Set background color to white

% Create a new figure for the decoder scatter plot
figure;
axis square;
hold on; % Allows multiple plots to be overlaid

% Plot each neuron's decoder weights with colors from the heatmap
for i = 1:120
    % Map neuron index to a color in the colormap
    colorIdx = round(1 + (size(cmap, 1) - 1) * (i - 1) / 119);
    scatter(i * ones(size(decoder_weights, 2), 1), decoder_weights(i, :), ...
        'filled', 'CData', cmap(colorIdx, :), 'SizeData', markerSize);
end

hold off;
title('Decoder Weights Comparison', 'FontSize', 50, 'FontWeight', 'bold');
xlabel('Neurons index', 'FontSize', 50, 'FontWeight', 'bold');
ylabel('Decoder weights', 'FontSize', 50, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold'); % Set axes tick label properties
grid on;
box on;
set(gcf, 'Color', 'w'); % Set background color to white



%% (8) Fit plot
% Fit plot between original matrix and reconstructed matrix
figure;
scatter(orginal_data(:, i), reconstructed_matrix(:, i), 'Marker', 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue', 'SizeData', 600); % Adjust SizeData for marker size
hold on;

% Calculate and plot the fit line
fit_line = polyfit(orginal_data(:, i), reconstructed_matrix(:, i), 1);
x_fit = linspace(min(orginal_data(:, i)), max(orginal_data(:, i)), 100); % For smoother line
y_fit = polyval(fit_line, x_fit);
plot(x_fit, y_fit, 'r-', 'LineWidth', 15); % Set line width to 4

% Calculate R-squared value
correlation = corrcoef(orginal_data(:, i), reconstructed_matrix(:, i));
r_squared = correlation(1, 2)^2;

% Display R-squared value on the plot
text(min(orginal_data(:, i)), max(reconstructed_matrix(:, i)), ['R^2 = ' num2str(r_squared)], 'FontSize', 50, 'FontWeight', 'bold'); % Set R-squared text size to 40 and bold

hold off;

xlabel('-Im(Z)/\Omega and Re(Z)/\Omega ↓', 'FontSize', 55, 'FontWeight', 'bold', 'Interpreter', 'tex');
ylabel('Reconstructed', 'FontSize', 55, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold'); % Set axes tick label properties
grid on;
box on;
axis square;


%% MSE and RMSE

% If your vectors are not one-dimensional, flatten them
test_vector_flat = orginal_data(:);
reconstructed_test_vector_flat = reconstructed_matrix(:);

% Calculate MSE
mse_value = mean((test_vector_flat - reconstructed_test_vector_flat).^2);

% Calculate RMSE
rmse_value = sqrt(mse_value);

% Create a table
results_table = table(mse_value, rmse_value, 'VariableNames', {'MSE', 'RMSE'});

% Display the table
disp(results_table);




%% Features as images

% Number of columns to display as images
num_columns_to_display = 5;

% Number of rows (features) in each column
num_rows = size(feature_matrix, 1);  % Assuming 'feature_matrix' has features as rows

% Create a figure window
figure;

% Loop through the first five columns
for i = 1:num_columns_to_display
    % Select subplot for the i-th column
    subplot(1, num_columns_to_display, i);
    
    % Extract the i-th column from feature_matrix
    feature_image = feature_matrix(:, i);
    
    % Since the column is a 1D vector, reshape it into a 2D matrix.
    % If 'num_rows' is not a perfect square, you need to define the
    % dimensions such that num_rows_in_image x num_columns_in_image = num_rows.
    % For this example, let's assume it's reshaped into a square for simplicity.
    side_length = ceil(sqrt(num_rows));  % Calculate the side length of the square
    padded_length = side_length ^ 2;     % Total elements in the square matrix
    
    % Pad the feature column with zeros if necessary
    feature_image_padded = [feature_image; zeros(padded_length - num_rows, 1)];
    
    % Reshape the padded column into a 2D matrix
    feature_image_2D = reshape(feature_image_padded, [side_length, side_length]);
    
    % Display the image
    imagesc(feature_image_2D);
    colormap('gray');  % Optional: specify a colormap
    colorbar;          % Show a colorbar
    
    % Add a title
    title(sprintf('Column %d as Image', i));
    
    % Remove axis labels
    axis off;
end

% Adjust layout to prevent subplots from overlapping, if necessary
sgtitle('First 5 Columns of Feature Matrix as Images');  % Add a main title for the figure

