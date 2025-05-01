

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 25C01 ,  , and 35 , 45, 25C05 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data_Capacity

filename = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\Capacity data\Data_Capacity_45C02.txt';
dataTable = readtable(filename);
data_capacity = table2array(dataTable);



A = data_capacity(:,2);
B = data_capacity(:,4);

Q_nominal = max(B(A == 0));   % Nominal charge capacity in mA.h --> its max capacity in cycle 0

[C, ~, ic] = unique(A);
B = accumarray(ic, B, [], @max);

SOH = (B / Q_nominal) * 100;
A=1:length(SOH);


figure;
plot(A, SOH, 'LineWidth', 4, 'Color', 'black');
xlabel('Cycle number', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('SOH (%)', 'FontSize', 40, 'FontWeight', 'bold');
set(gca, 'FontSize', 40, 'FontWeight', 'bold', 'LineWidth', 4);  % Set font size, weight of the tick labels, and box line width

grid on;
box on;

figure;
plot(A, B, 'LineWidth', 4, 'Color', 'black');
xlabel('Cycle number', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('Capacity (mAh)', 'FontSize', 40, 'FontWeight', 'bold');
set(gca, 'FontSize', 40, 'FontWeight', 'bold', 'LineWidth', 4);  % Set font size, weight of the tick labels, and box line width

grid on;
box on;

%% Data_EIS

filename = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_45C02.txt';
dataTable = readtable(filename);
data_EIS = table2array(dataTable);


%% Extract EIS data for 90%, 80%, 70%, and 60% SOH and plot

percentages = [100,90, 80, 50];
colors = {'black', 'blue', 'red', 'green'};
figure;

for i = 1:length(percentages)
    p = percentages(i);
    [~, idx] = min(abs(SOH - p)); % Find index of value closest to the percentage
    cycle_number = A(idx); % Extract cycle number

    indices = find(data_EIS(:, 2) == cycle_number); % Find indices of the cycle number in the EIS data
    plot(data_EIS(indices, 4), data_EIS(indices, 5), 'LineStyle', '-', 'Marker', 'p', 'DisplayName', strcat(num2str(p), '% SOH'), 'LineWidth', 4, 'Color', colors{i});
    hold on;
end

xlabel('Re(Z)/Ohm', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('-Im(Z)/Ohm', 'FontSize', 40, 'FontWeight', 'bold');
legend('FontSize', 40);
set(gca, 'FontSize', 40, 'FontWeight', 'bold', 'LineWidth', 4);  % Set font size, weight of the tick labels, and box line width

grid on;
box on;

%% Extract input and output

% Input % Each raw in Input represent (60) Re(Z)/Ohm + (60) -Im(Z)/Ohm for each cycle 

% Determine the number of full groups of 60 in column 2
num_groups = floor(size(data_EIS, 1) / 60);

% Initialize the Input matrix
Input = zeros(120, num_groups);

for i = 1:num_groups
    start_idx = (i-1)*60 + 1;
    end_idx = i*60;
    
    % Extract data from columns 4 and 5 and flatten them into a 120-row vector
    col_vector = [data_EIS(start_idx:end_idx, 4); data_EIS(start_idx:end_idx, 5)];
    
    % Store this vector in the Input matrix
    Input(:, i) = col_vector;
end

% Output

Output=B; % output is capacity %



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 25C02 , 25C03 , 25C04 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data_Capacity

filename = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\Capacity data\Data_Capacity_25C02.txt';
dataTable = readtable(filename);
data_capacity = table2array(dataTable);


A = data_capacity(:,2);
B = data_capacity(:,6);

Q_nominal = max(B(A == 0));   % Nominal charge capacity in mA.h --> its max capacity in cycle 0

[C, ~, ic] = unique(A);
B = accumarray(ic, B, [], @max);

SOH = (B / Q_nominal) * 100;
A=1:length(SOH);


figure;
plot(A, SOH, 'LineWidth', 4, 'Color', 'black');
xlabel('Cycle number', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('SOH (%)', 'FontSize', 40, 'FontWeight', 'bold');
set(gca, 'FontSize', 40, 'FontWeight', 'bold', 'LineWidth', 4);  % Set font size, weight of the tick labels, and box line width

grid on;
box on;

figure;
plot(A, B, 'LineWidth', 4, 'Color', 'black');
xlabel('Cycle number', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('Capacity (mAh)', 'FontSize', 40, 'FontWeight', 'bold');
set(gca, 'FontSize', 40, 'FontWeight', 'bold', 'LineWidth', 4);  % Set font size, weight of the tick labels, and box line width

grid on;
box on;

%% Data_EIS

filename = 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Datasets\EIS data\EIS_state_I_25C02.txt';
dataTable = readtable(filename);
data_EIS = table2array(dataTable);



%% Extract EIS data for 90%, 80%, 70%, and 60% SOH and plot

percentages = [100,90, 80, 50];
colors = {'black', 'blue', 'red', 'green'};
figure;

for i = 1:length(percentages)
    p = percentages(i);
    [~, idx] = min(abs(SOH - p)); % Find index of value closest to the percentage
    cycle_number = A(idx); % Extract cycle number

    indices = find(data_EIS(:, 2) == cycle_number); % Find indices of the cycle number in the EIS data
    plot(data_EIS(indices, 4), data_EIS(indices, 5), 'LineStyle', '-', 'Marker', 'p', 'DisplayName', strcat(num2str(p), '% SOH'), 'LineWidth', 4, 'Color', colors{i});
    hold on;
end

xlabel('Re(Z)/Ohm', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('-Im(Z)/Ohm', 'FontSize', 40, 'FontWeight', 'bold');
legend('FontSize', 40);
set(gca, 'FontSize', 40, 'FontWeight', 'bold', 'LineWidth', 4);  % Set font size, weight of the tick labels, and box line width

grid on;
box on;

%% Extract input and output

% Input % Each raw in Input represent (60) Re(Z)/Ohm + (60) -Im(Z)/Ohm for each cycle 

% Determine the number of full groups of 60 in column 2
num_groups = floor(size(data_EIS, 1) / 60);

% Initialize the Input matrix
Input2 = zeros(120, num_groups);

for i = 1:num_groups
    start_idx = (i-1)*60 + 1;
    end_idx = i*60;
    
    % Extract data from columns 4 and 5 and flatten them into a 120-row vector
    col_vector = [data_EIS(start_idx:end_idx, 4); data_EIS(start_idx:end_idx, 5)];
    
    % Store this vector in the Input matrix
    Input2(:, i) = col_vector;
end

% Output

Output2=B; % output is capacity %







