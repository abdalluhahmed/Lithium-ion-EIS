import os
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1) Load data
mat = scipy.io.loadmat(
    r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\Dataset-State_I\Training\Inputs_All.mat"
)
Inputs_All = mat["Inputs_All"]
original_data = Inputs_All.copy()

# 2) Scale to zero mean/unit variance
scaler = StandardScaler()
X = scaler.fit_transform(original_data.T)   # shape: (n_signals, 120)

# 3) Build sparse autoencoder
input_dim = X.shape[1]
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

# 4) Checkpoint: save only best
ckpt_dir = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\run-in-python\checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
best_ckpt = os.path.join(ckpt_dir, "autoencoder-best.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_ckpt,
    monitor="loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# 5) Train
history = autoencoder.fit(
    X, X,
    epochs=10000,
    batch_size=X.shape[0],
    verbose=1,
    callbacks=[checkpoint_cb]
)

# 6) Reload best & save encoder weights
autoencoder.load_weights(best_ckpt)
encoder.save_weights(os.path.join(ckpt_dir, "encoder-best.h5"))

# 7) Encode & decode all signals
encoded_data   = encoder.predict(X)            # (n_signals, 20)
decoded_data   = autoencoder.predict(X)        # (n_signals, 120)
reconstructed  = scaler.inverse_transform(decoded_data).T
feature_matrix = encoded_data.T

# 8) Plot original / features / reconstructed
segment_size = input_dim // 2   # 60
rows_per_fig = 3
column_ranges = [
    range(0, 261),
    range(261, 443),
    range(443, 664),
    range(664, 681),
    range(681, 1007),
    range(1007, original_data.shape[1]),
]
titles = ["25C01", "25C02", "25C03", "25C04", "35C01", "45C01"]
axis_font = dict(fontsize=22, fontweight="bold")
tick_fontsize = 20

for fig_idx in range(2):
    plt.figure(figsize=(18, 10))
    for row in range(rows_per_fig):
        seg_idx = fig_idx * rows_per_fig + row
        if seg_idx >= len(column_ranges):
            break
        cols = column_ranges[seg_idx]
        cmap = plt.cm.jet(np.linspace(0, 1, len(cols)))
        for sub in range(5):
            left = 0.1 + sub * 0.18
            bottom = 1 - (row + 1) * 0.29 + 0.02
            ax = plt.gcf().add_axes([left, bottom, 0.1, 0.2])
            if sub < 2:
                start = sub * segment_size
                data = original_data[start:start + segment_size, cols]
            elif sub == 2:
                data = feature_matrix[:, cols]
            else:
                start = (sub - 3) * segment_size
                data = reconstructed[start:start + segment_size, cols]
            for i, c in enumerate(cols):
                ax.plot(data[:, i], linewidth=4, color=cmap[i])
            ax.grid(True)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == rows_per_fig // 2:
                if sub == 0:
                    ax.set_ylabel("Re(Z)/Ω", **axis_font)
                elif sub == 1:
                    ax.set_ylabel("-Im(Z)/Ω", **axis_font)
                elif sub == 2:
                    ax.set_ylabel("Features", **axis_font)
                    ax.set_xlabel("Feature Frequencies", **axis_font)
                else:
                    ax.set_ylabel("Reconstructed", **axis_font)
            if sub == 0:
                ax.set_title(titles[seg_idx], **axis_font)
            ax.tick_params(labelsize=tick_fontsize)
    plt.show()

# 9) Plot encoder weights (first 5 neurons)
enc_w = autoencoder.get_layer("encoded").get_weights()[0]  # shape (120,20)
plt.figure(figsize=(16, 4))
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    ax.bar(range(input_dim), enc_w[:, i])
    ax.set_title(f"Encoder Weights – Neuron {i+1}", fontsize=14)
    ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()

# 10) Plot decoder weights (first 5 outputs)
dec_w = autoencoder.get_layer("decoded").get_weights()[0]  # shape (20,120)
plt.figure(figsize=(16, 4))
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    ax.bar(range(hidden_dim), dec_w[:, i])
    ax.set_title(f"Decoder Weights – Output {i+1}", fontsize=14)
    ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()

# 11) Scatter heatmap style
cmap_full = plt.cm.jet(np.linspace(0, 1, 64))
plt.figure(figsize=(12,12))
for i in range(hidden_dim):
    color = cmap_full[int(i*(63/(hidden_dim-1)))]
    y = enc_w[:, i]
    x = np.full_like(y, i+1)
    plt.scatter(x, y, s=140, c=[color], marker='o')
plt.title("Encoder Weights Comparison", fontsize=20, fontweight="bold")
plt.xlabel("Neuron Index", fontsize=18, fontweight="bold")
plt.ylabel("Weight", fontsize=18, fontweight="bold")
plt.grid(True)
plt.box(True)
plt.show()

plt.figure(figsize=(12,12))
for i in range(input_dim):
    color = cmap_full[int(i*(63/(input_dim-1)))]
    y = dec_w[:, i]
    x = np.full_like(y, i+1)
    plt.scatter(x, y, s=140, c=[color], marker='o')
plt.title("Decoder Weights Comparison", fontsize=20, fontweight="bold")
plt.xlabel("Output Index", fontsize=18, fontweight="bold")
plt.ylabel("Weight", fontsize=18, fontweight="bold")
plt.grid(True)
plt.box(True)
plt.show()

# 12) Fit plot for first signal
j = 0
plt.figure(figsize=(8,8))
plt.scatter(original_data[:,j], reconstructed[:,j], s=600, alpha=0.6, edgecolor='k')
m, b = np.polyfit(original_data[:,j], reconstructed[:,j], 1)
x_fit = np.linspace(original_data[:,j].min(), original_data[:,j].max(), 200)
plt.plot(x_fit, m*x_fit+b, 'r-', linewidth=4)
r2 = np.corrcoef(original_data[:,j], reconstructed[:,j])[0,1]**2
plt.text(0.05, 0.95, f"$R^2={r2:.3f}$", transform=plt.gca().transAxes, fontsize=18, verticalalignment='top')
plt.xlabel("-Im(Z)/Ω and Re(Z)/Ω ↓", fontsize=20, fontweight='bold')
plt.ylabel("Reconstructed", fontsize=20, fontweight='bold')
plt.grid(True)
plt.axis('square')
plt.show()

# 13) MSE & RMSE
err = original_data - reconstructed
mse = np.mean(err**2)
rmse = np.sqrt(mse)
print(f"MSE = {mse:.6e},   RMSE = {rmse:.6e}")

# 14) First 5 feature columns as images
side = int(np.ceil(np.sqrt(hidden_dim)))
pad_len = side**2
plt.figure(figsize=(15,3))
for i in range(5):
    vec = feature_matrix[:, i]
    padded = np.hstack([vec, np.zeros(pad_len - hidden_dim)])
    img = padded.reshape(side, side)
    ax = plt.subplot(1, 5, i+1)
    ax.imshow(img, aspect='equal', cmap='gray')
    ax.set_title(f"Col {i+1} as Image")
    ax.axis('off')
plt.suptitle("First 5 Columns of Feature Matrix as Images")
plt.show()
