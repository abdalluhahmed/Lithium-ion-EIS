
function selectedValuesCell = excludeOutliersAndMean(errorArray)
    numDataPoints = size(errorArray, 1); % Number of rows (data points)
    selectedValuesCell = cell(1, numDataPoints); % Initialize cell array

    for i = 1:numDataPoints
        data = errorArray(i, :); % Extract errors for the ith data point (row)

        % Define the threshold for close values
        threshold = 50 * std(data);

        % Find values that are close to their median value
        closeValues = data(abs(data - median(data)) < threshold);

        % Store the selected close values in the cell array
        selectedValuesCell{i} = closeValues;
    end
end