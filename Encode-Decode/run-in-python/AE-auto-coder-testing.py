import os
import numpy as np
import scipy.io
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# --- Paths ---
train_mat_path = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Training\Inputs_All.mat"
ckpt_dir       = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\run-in-python\checkpoints"
features_dir   = os.path.join(ckpt_dir, "Features")
os.makedirs(features_dir, exist_ok=True)

best_autoenc   = os.path.join(ckpt_dir, "autoencoder-best.h5")
best_encoder   = os.path.join(ckpt_dir, "encoder-best.h5")

test35_mat     = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Testing\Input_35C02.mat"
test45_mat     = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Testing\Input_45C02.mat"

# --- 1) Reload training data & refit scaler ---
train_data = loadmat(train_mat_path).get("Inputs_All")
if train_data is None:
    raise FileNotFoundError(f"'Inputs_All' not found in {train_mat_path}")
scaler = StandardScaler()
_ = scaler.fit(train_data.T)

# --- 2) Rebuild the autoencoder architecture ---
input_dim = train_data.shape[0]  # 120
hidden_dim = 20

inp = Input(shape=(input_dim,), name="encoder_input")
encoded = Dense(
    hidden_dim,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.00004),
    activity_regularizer=regularizers.l1(0.004),
    name="encoded",
)(inp)
decoded = Dense(
    input_dim,
    activation="linear",
    kernel_regularizer=regularizers.l2(0.00004),
    name="decoded",
)(encoded)

autoencoder = Model(inputs=inp, outputs=decoded, name="autoencoder")
encoder     = Model(inputs=inp, outputs=encoded, name="encoder")
autoencoder.compile(optimizer="adam", loss="mse")

# --- 3) Load the best weights ---
autoencoder.load_weights(best_autoenc)
encoder.load_weights(best_encoder)

# --- 3.1) Encode and save training features ---
train_scaled        = scaler.transform(train_data.T)       # (n_signals, 120)
features_all        = encoder.predict(train_scaled)        # (n_signals, 20)
feature_matrix_all  = features_all.T                       # (20, n_signals)
savemat(
    os.path.join(features_dir, "feature_matrix.mat"),
    {"feature_matrix": feature_matrix_all}
)
print(f"Saved training feature_matrix to {os.path.join(features_dir, 'feature_matrix.mat')}")

# --- 4) Load both test datasets ---
mat35 = loadmat(test35_mat)
mat45 = loadmat(test45_mat)
Input_35C02 = mat35.get("Input_35C02")  # (120, n35)
Input_45C02 = mat45.get("Input_45C02")  # (120, n45)
if Input_35C02 is None or Input_45C02 is None:
    raise FileNotFoundError("Test inputs not found in specified .mat files.")

# --- 5) Preprocess both test vectors ---
test35_scaled = scaler.transform(Input_35C02.T)  # (n35, 120)
test45_scaled = scaler.transform(Input_45C02.T)  # (n45, 120)

# --- 6) Encode both and reconstruct ---
features_35 = encoder.predict(test35_scaled)      # (n35, 20)
decoded_35  = autoencoder.predict(test35_scaled)  # (n35, 120)
features_45 = encoder.predict(test45_scaled)      # (n45, 20)
decoded_45  = autoencoder.predict(test45_scaled)  # (n45, 120)

# --- 7) Convert back to original scale ---
reconstructed_35  = scaler.inverse_transform(decoded_35).T  # (120, n35)
reconstructed_45  = scaler.inverse_transform(decoded_45).T  # (120, n45)
feature_matrix_35 = features_35.T                         # (20, n35)
feature_matrix_45 = features_45.T                         # (20, n45)

# --- 8) Save test features to .mat files ---
savemat(
    os.path.join(features_dir, "features_test_35C02.mat"),
    {"features_test_35C02": feature_matrix_35}
)
savemat(
    os.path.join(features_dir, "features_test_45C02.mat"),
    {"features_test_45C02": feature_matrix_45}
)
print(f"Saved test feature matrices to {features_dir}")

# --- 9) Save combined CSV of 35 & 45 features ---
f35 = feature_matrix_35
f45 = feature_matrix_45
features_35_45 = np.hstack([f35, f45])  # shape (20, 2) if one signal each
csv_path = os.path.join(features_dir, "features_35_45.csv")
np.savetxt(
    csv_path,
    features_35_45,
    delimiter=",",
    header="35C02,45C02",
    comments=""
)
print(f"Saved combined CSV to {csv_path}")

# --- 10) Example plotting for 45C02 ---

# Original halves
num_signals = Input_45C02.shape[1]
cm = plt.cm.jet(np.linspace(0, 1, num_signals))

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
for i in range(num_signals):
    plt.plot(Input_45C02[:60, i], linewidth=4, color=cm[i])
plt.title('Original 45C02: Re(Z)/Ω', fontsize=20)
plt.xlabel('Frequencies', fontsize=18)
plt.ylabel('Re(Z)/Ω', fontsize=18)
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(num_signals):
    plt.plot(Input_45C02[60:, i], linewidth=4, color=cm[i])
plt.title('Original 45C02: -Im(Z)/Ω', fontsize=20)
plt.xlabel('Frequencies', fontsize=18)
plt.ylabel('-Im(Z)/Ω', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# Encoded features
plt.figure(figsize=(8, 8))
for i in range(num_signals):
    plt.plot(feature_matrix_45[:, i], linewidth=4, color=cm[i])
plt.title('Encoded Features 45C02', fontsize=20)
plt.xlabel('Feature Index', fontsize=18)
plt.ylabel('Feature Value', fontsize=18)
plt.grid(True)
plt.show()

# Reconstructed halves
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
for i in range(num_signals):
    plt.plot(reconstructed_45[:60, i], linewidth=4, color=cm[i])
plt.title('Reconstructed 45C02: Re(Z)/Ω', fontsize=20)
plt.xlabel('Frequencies', fontsize=18)
plt.ylabel('Reconstructed', fontsize=18)
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(num_signals):
    plt.plot(reconstructed_45[60:, i], linewidth=4, color=cm[i])
plt.title('Reconstructed 45C02: -Im(Z)/Ω', fontsize=20)
plt.xlabel('Frequencies', fontsize=18)
plt.ylabel('Reconstructed', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit plot & R²
j = 0
plt.figure(figsize=(8, 8))
plt.scatter(Input_45C02[:, j], reconstructed_45[:, j], s=100, alpha=0.6)
m, b = np.polyfit(Input_45C02[:, j], reconstructed_45[:, j], 1)
x_fit = np.linspace(Input_45C02[:, j].min(), Input_45C02[:, j].max(), 200)
plt.plot(x_fit, m*x_fit + b, 'r--', linewidth=3)
r2 = np.corrcoef(Input_45C02[:, j], reconstructed_45[:, j])[0, 1]**2
plt.text(0.05, 0.95, f'$R^2={r2:.3f}$', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
plt.title('Fit: 45C02', fontsize=20)
plt.xlabel('Original', fontsize=18)
plt.ylabel('Reconstructed', fontsize=18)
plt.grid(True)
plt.axis('square')
plt.show()

# MSE & RMSE
mse_val = np.mean((Input_45C02 - reconstructed_45)**2)
rmse_val = np.sqrt(mse_val)
print(f"45C02 MSE = {mse_val:.6e}, RMSE = {rmse_val:.6e}")
