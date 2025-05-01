import numpy as np
import scipy.io
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as pl


# Load your dataset
inputs_all = scipy.io.loadmat('C:/Users/msi-pc/Desktop/Li-ion paper 2/Model/Encode-Decode/Dataset-State_V/Training/Inputs_All.mat')['Inputs_All']
output_all = scipy.io.loadmat('C:/Users/msi-pc/Desktop/Li-ion paper 2/Model/Encode-Decode/Dataset-State_V/Training/Output_All.mat')['Output_All']
input_35C02 = scipy.io.loadmat('C:/Users/msi-pc/Desktop/Li-ion paper 2/Model/Encode-Decode/Dataset-State_V/Testing/Input_35C02.mat')['Input_35C02']
output_35C02 = scipy.io.loadmat('C:/Users/msi-pc/Desktop/Li-ion paper 2/Model/Encode-Decode/Dataset-State_V/Testing/Output_35C02.mat')['Output_35C02']

# Preprocessing
# Reshape output_all if it's not a 1D array
if output_all.ndim > 1:
    output_all = output_all.ravel()

# Preprocessing
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs_all.T)

# Training the SVR model
svr = SVR(kernel='rbf', C=3000, epsilon=0.1)
svr.fit(inputs_scaled, output_all)

# Normalize the testing data and make predictions
input_35C02_scaled = scaler.transform(input_35C02.T)
output_35C02_pred = svr.predict(input_35C02_scaled)

# Calculate SOH for actual and predicted data
SOH_output_35C02 = (output_35C02 / output_35C02[0]) * 100
SOH_output_35C02_pred = (output_35C02_pred / output_35C02_pred[0]) * 100

# Define variability percentage for confidence interval
# Assuming SOH_output_35C02 and SOH_output_35C02_pred are defined as before
# Define variability percentage for confidence interval
variability_percentage = 0.03
upper_bound = SOH_output_35C02_pred * (1 + variability_percentage)
lower_bound = SOH_output_35C02_pred * (1 - variability_percentage)
# X-axis values similar to MATLAB's [2:2:598]
x_values = range(2, 599, 2)
# Plotting
plt.figure(figsize=(15, 8))
# Filling the area for the confidence interval
plt.fill_between(x_values, upper_bound[:len(x_values)], lower_bound[:len(x_values)], color='red', alpha=0.1)
# Plotting the actual and predicted SOH
p1, = plt.plot(x_values, SOH_output_35C02[:len(x_values)], 'b-*', linewidth=1, markersize=10)
p2, = plt.plot(x_values, SOH_output_35C02_pred[:len(x_values)], 'r-*', linewidth=1, markersize=10)
# Setting plot limits and labels
plt.xlim([0, 400])
plt.ylim([70, 100])
plt.xlabel('Cycle number', fontsize=20, fontweight='bold')
plt.ylabel('SOH (%)', fontsize=20, fontweight='bold')
plt.title('35C02', fontsize=20, fontweight='bold')
plt.legend([p1, p2], ['Actual SOH (35C02)', 'Estimated SOH (35C02)'], fontsize=16)
plt.grid(True)
plt.show()

# Assuming SOH_output_35C02_pred is already calculated

# Print the values of SOH_output_35C02_pred
print("Values of SOH_output_35C02_pred:")
for value in SOH_output_35C02_pred:
    print(value)



#################

SOH_output_35C02 = (output_35C02 / output_35C02[0])
SOH_output_35C02_pred = (output_35C02_pred / output_35C02_pred[0])



# ME - Mean Error
mean_error = np.mean(SOH_output_35C02_pred - SOH_output_35C02)

# MSE - Mean Squared Error
mse = mean_squared_error(SOH_output_35C02, SOH_output_35C02_pred)

# RMSE - Root Mean Squared Error
rmse = np.sqrt(mse)

# R^2 - Coefficient of Determination
r2 = r2_score(SOH_output_35C02, SOH_output_35C02_pred)

# MAPE - Mean Absolute Percentage Error
mape = np.mean(np.abs((SOH_output_35C02 - SOH_output_35C02_pred) / SOH_output_35C02))

# Print the results
print(f'Mean Error (ME): {mean_error:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R^2): {r2:.4f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
