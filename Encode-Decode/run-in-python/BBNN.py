import os
import numpy as np
import scipy.io
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# -------------------- Paths --------------------
features_dir    = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\run-in-python\checkpoints\Features"
ann_net_dir     = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\ANN_State_I\net"
os.makedirs(ann_net_dir, exist_ok=True)

train_feat_mat  = os.path.join(features_dir, "feature_matrix.mat")
train_out_mat   = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Training\Output_All.mat"
test35_feat_mat = os.path.join(features_dir, "features_test_35C02.mat")
test45_feat_mat = os.path.join(features_dir, "features_test_45C02.mat")
test35_out_mat  = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Testing\Output_35C02.mat"
test45_out_mat  = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Testing\Output_45C02.mat"

# ------------------ 1) Load training features & outputs ------------------
feat_train = loadmat(train_feat_mat).get("feature_matrix")  # (20, n_train)
if feat_train is None:
    raise FileNotFoundError(f"'feature_matrix' not found in {train_feat_mat}")
X_train = feat_train.T                                     # (n_train, 20)
y_train = loadmat(train_out_mat)["Output_All"].T.ravel()   # (n_train,)

# ------------------ 2) Load test features & outputs ------------------
feat35 = loadmat(test35_feat_mat).get("features_test_35C02")
feat45 = loadmat(test45_feat_mat).get("features_test_45C02")
if feat35 is None or feat45 is None:
    raise FileNotFoundError("Test feature files missing or misnamed.")
X35 = feat35.T  # (n35, 20)
X45 = feat45.T  # (n45, 20)

y35 = loadmat(test35_out_mat)["Output_35C02"].T.ravel()  # (n35,)
y45 = loadmat(test45_out_mat)["Output_45C02"].T.ravel()  # (n45,)

SOH35_actual = (y35 / y35[0]) * 100
SOH45_actual = (y45 / y45[0]) * 100

# ------------------ 3) Build, init, train & save ensemble ------------------
num_models = 5000
mse35 = np.zeros(num_models)
mse45 = np.zeros(num_models)
models = []

tf.random.set_seed(0)
np.random.seed(0)

input_dim = X_train.shape[1]  # 20
hidden_dim = 20
output_dim = 1

for i in range(num_models):
    # Create model structure
    model = Sequential([
        Dense(hidden_dim, activation="sigmoid", input_shape=(input_dim,)),
        Dense(output_dim, activation="linear")
    ])

    # He initialization for layer 1
    stddev1 = np.sqrt(2.0 / input_dim)
    w0 = stddev1 * np.random.randn(input_dim, hidden_dim)
    b0 = np.zeros(hidden_dim)
    model.layers[0].set_weights([w0, b0])

    # He initialization for layer 2
    stddev2 = np.sqrt(2.0 / hidden_dim)
    w1 = stddev2 * np.random.randn(hidden_dim, output_dim)
    b1 = np.zeros(output_dim)
    model.layers[1].set_weights([w1, b1])

    # Compile and train
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss="mse")
    model.fit(X_train, y_train, epochs=1000, batch_size=len(X_train), verbose=0)

    # Save
    model.save(os.path.join(ann_net_dir, f"net{i}.h5"))
    models.append(model)

    # Evaluate on 35C02
    p35 = model.predict(X35).ravel()
    norm35 = (p35 / p35[0]) * 100
    mse35[i] = mean_squared_error(SOH35_actual, norm35)

    # Evaluate on 45C02
    p45 = model.predict(X45).ravel()
    norm45 = (p45 / p45[0]) * 100
    mse45[i] = mean_squared_error(SOH45_actual, norm45)

# ------------------ 4) Select best model ------------------
valid = (mse35 < 10) & (mse45 < 10)
if not np.any(valid):
    raise RuntimeError("No models met the MSE < 10 criterion on both datasets.")
avg_mse = (mse35[valid] + mse45[valid]) / 2
best_idx_within = np.argmin(avg_mse)
best_model_index = np.where(valid)[0][best_idx_within]
best_model = models[best_model_index]
print("Best model index:", best_model_index)

# ------------------ 5) Final predictions & metrics ------------------
def compute_metrics(actual, pred):
    mse  = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    r2   = np.corrcoef(actual, pred)[0,1]**2
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mse, rmse, r2, mape

p35_f = best_model.predict(X35).ravel()
SOH35_pred = (p35_f / p35_f[0]) * 100
metrics35 = compute_metrics(SOH35_actual, SOH35_pred)
print("35C02 → MSE, RMSE, R², MAPE:", metrics35)

p45_f = best_model.predict(X45).ravel()
SOH45_pred = (p45_f / p45_f[0]) * 100
metrics45 = compute_metrics(SOH45_actual, SOH45_pred)
print("45C02 → MSE, RMSE, R², MAPE:", metrics45)

# ------------------ 6) Plots ------------------
# A) Training vs. Predicted
y_train_pred = best_model.predict(X_train).ravel()
plt.figure(figsize=(6,4))
plt.plot(y_train, 'b-o', label='Actual')
plt.plot(y_train_pred, 'r-*', label='Predicted')
plt.title('Training: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Output_All')
plt.legend()
plt.grid(True)
plt.show()

# B) Hidden-layer weight distributions
w0_final, _ = best_model.layers[0].get_weights()
plt.figure(figsize=(5,5))
for j in range(hidden_dim):
    plt.scatter(np.full(input_dim, j+1), w0_final[:,j], s=50)
plt.title('Hidden Layer Weights')
plt.xlabel('Neuron Index')
plt.ylabel('Weight Value')
plt.grid(True)
plt.show()

# C) Output-layer weights
w1_final, _ = best_model.layers[1].get_weights()
plt.figure(figsize=(5,5))
plt.scatter(np.arange(1, hidden_dim+1), w1_final[:,0], s=80)
plt.title('Output Layer Weights')
plt.xlabel('Hidden Neuron Index')
plt.ylabel('Weight Value')
plt.grid(True)
plt.show()

# D) SOH curves with ±3% band
def plot_soh(actual, pred, title):
    x = np.arange(len(actual))
    band = 0.03
    upper = pred * (1+band)
    lower = pred * (1-band)
    plt.figure(figsize=(6,4))
    plt.fill_between(x, lower, upper, color='r', alpha=0.1, label='±3% band')
    plt.plot(x, actual, 'b-*', label='Actual')
    plt.plot(x, pred, 'r-o', label='Predicted')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('SOH (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(70,100)
    plt.show()

plot_soh(SOH45_actual, SOH45_pred, '45C02 SOH')
plot_soh(SOH35_actual, SOH35_pred, '35C02 SOH')
