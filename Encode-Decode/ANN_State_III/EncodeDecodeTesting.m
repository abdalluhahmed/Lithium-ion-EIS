





load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_III\Testing\Input_35C02.mat'

load 'C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_III\Testing\Input_45C02.mat'


%%
% Assuming dQdV is your original vector of size 1000x1
test_vector=Input_45C02;

% test_vector(56:60,:)=[];
% test_vector(113:end,:)=[];



% Encode the extended test vector using the trained autoencoder
features_test = encode(autoenc, test_vector);

% Reconstruct the extended test vector
reconstructed_test_vector = decode(autoenc, features_test);

%% Create a figure and use subplot to create a layout for the plots
figure;
axis square;
cm = jet(size(test_vector, 2));  % Store jet colormap

% Subplot for the first half of the test vector
subplot(1, 2, 1);
hold on;
for i = 1:size(test_vector, 2)
    plot(test_vector(1:60, i), 'LineWidth', 4, 'Color', cm(i, :));
end
hold off;
ylabel('Re(Z)/\Omega', 'FontSize', 55, 'FontWeight', 'bold');
xlabel('Frequencies', 'FontSize', 55, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold', 'Box', 'on');
grid on;

% Subplot for the second half of the test vector
subplot(1, 2, 2);
hold on;
for i = 1:size(test_vector, 2)
    plot(test_vector(61:end, i), 'LineWidth', 4, 'Color', cm(i, :));
end
hold off;
ylabel('-Im(Z)/\Omega', 'FontSize', 55, 'FontWeight', 'bold');
xlabel('Frequencies', 'FontSize', 55, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold', 'Box', 'on');
grid on;

% Create a separate figure for the plot of the features
figure;
axis square;
cm = jet(size(features_test, 2));  % Store jet colormap for features
hold on;
for i = 1:size(features_test, 2)
    plot(features_test(:, i), 'LineWidth', 4, 'Color', cm(i, :));
end
hold off;
ylabel('Features', 'FontSize', 55, 'FontWeight', 'bold');
xlabel('Frequencies', 'FontSize', 55, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold', 'Box', 'on');
grid on;

% Create a separate figure for the plot of the reconstructed test vector
figure;
axis square;
cm = jet(size(reconstructed_test_vector, 2));  % Store jet colormap for reconstructed data

% Subplot for the first half of the reconstructed test vector
subplot(1, 2, 1);
hold on;
for i = 1:size(reconstructed_test_vector, 2)
    plot(reconstructed_test_vector(1:60, i), 'LineWidth', 4, 'Color', cm(i, :));
end
hold off;
ylabel('Reconstructed', 'FontSize', 55, 'FontWeight', 'bold');
xlabel('Frequencies', 'FontSize', 55, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold', 'Box', 'on');
grid on;

% Subplot for the second half of the reconstructed test vector
subplot(1, 2, 2);
hold on;
for i = 1:size(reconstructed_test_vector, 2)
    plot(reconstructed_test_vector(61:end, i), 'LineWidth', 4, 'Color', cm(i, :));
end
hold off;
ylabel('Reconstructed', 'FontSize', 55, 'FontWeight', 'bold');
xlabel('Frequencies', 'FontSize', 55, 'FontWeight', 'bold');
set(gca, 'FontSize', 50, 'FontWeight', 'bold', 'Box', 'on');
grid on;





%% Fit-plot
% Create a figure for the fit plot
figure;
scatter(test_vector, reconstructed_test_vector, 'Marker', 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue', 'SizeData', 800); % Adjust SizeData for marker size
hold on;
p = polyfit(test_vector, reconstructed_test_vector, 1); % Perform linear regression
plot(test_vector, polyval(p, test_vector), 'r-', 'LineWidth', 20); % Set line width to 4 for the fit line
hold off;
xlabel('-Im(Z)/Ω and Re(Z)/Ω ↓', 'FontSize', 55, 'FontWeight', 'bold'); % Update font size for x label
ylabel('Reconstructed', 'FontSize', 55, 'FontWeight', 'bold'); % Update font size for y label
set(gca, 'FontSize', 50, 'FontWeight', 'bold', 'Box', 'on'); % Update font size for plot elements
grid on;
axis square;

% Calculate R-squared
y_mean = mean(reconstructed_test_vector);
SS_total = sum((reconstructed_test_vector - y_mean).^2);
SS_residual = sum((reconstructed_test_vector - polyval(p, test_vector)).^2);
R_squared = 1 - SS_residual / SS_total;

% Display the R-squared value on the fit plot
text(0.1, 0.9, ['R^2 = ', num2str(R_squared)], 'Units', 'normalized', 'FontSize', 50, 'FontWeight', 'bold'); % Update font size for R-squared text




%%

% Assuming your test_vector and reconstructed_test_vector are already defined

% If your vectors are not one-dimensional, flatten them
test_vector_flat = test_vector(:);
reconstructed_test_vector_flat = reconstructed_test_vector(:);

% Calculate MSE
mse_value = mean((test_vector_flat - reconstructed_test_vector_flat).^2);

% Calculate RMSE
rmse_value = sqrt(mse_value);

% Create a table
results_table = table(mse_value, rmse_value, 'VariableNames', {'MSE', 'RMSE'});

% Display the table
disp(results_table);


