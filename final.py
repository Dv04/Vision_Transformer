###############################################################################
# 0.   GLOBAL SETTINGS – adjust if your paths differ
###############################################################################
IMG_H, IMG_W = 64, 64  # image size (matches saved model)
PATCH_SIZE = 8  # kept for ViT inference only
PCA_NCOMP = 20  # must match training
BATCH_SIZE = 32
VAL_BATCH_SIZE = 8
DATA_DIR = "data"  # root folder for images + CSV
VIT_H5_PATH = "iteration_5_model.h5"  # saved full model (inc. weights)
RESULT_DIR = "paper_results"  # all outputs land here

###############################################################################
# 1.   IMPORTS
###############################################################################
import os, glob, re, gc, cv2, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["GLOG_minloglevel"] = "3"  # Suppress GLOG messages
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC INFO and WARNING messages
os.environ["GRPC_CPP_LOG_LEVEL"] = "ERROR"  # Suppress gRPC C++ logs

import logging

logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf

tf.get_logger().setLevel("FATAL")

from tensorflow import keras
from keras import layers, regularizers
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    Reshape,
    Input,
)

from tensorflow.keras.callbacks import LambdaCallback

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Cell 1: Import Libraries

from sklearn.preprocessing import StandardScaler


os.makedirs(RESULT_DIR, exist_ok=True)


###############################################################################
# 2.   HELPERS  (splitter, augmentation, PCA, batch-gen) – 100 % compatible
###############################################################################


def rotate_img(im, angle):
    (h, w) = im.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(im, M, (nW, nH))


def weighted_mse(yTrue, yPred):
    # create a monotonically increasing weight vector
    ones = tf.ones_like(yTrue[0, :])
    idx = tf.math.cumsum(ones, axis=0)
    # apply weighted MSE
    loss = tf.reduce_mean((1.0 / tf.cast(idx, tf.float32)) * tf.square(yTrue - yPred))
    return loss


def collect_files():
    train, val, test = [], [], []
    for folder in glob.glob(f"{DATA_DIR}/*"):
        for fname in os.listdir(folder):
            if not fname.endswith(".jpg"):
                continue
            csv_path = os.path.join(folder, fname.replace(".jpg", ".csv"))
            if not os.path.exists(csv_path):
                continue
            try:
                row_count = pd.read_csv(csv_path).shape[0]
                if row_count != 205:
                    continue
            except Exception as e:
                print(f"Skipping {csv_path}: {e}")
                continue
            prob = random.random()
            fpath = os.path.join(folder, fname)
            if prob < 0.02:
                test.append(fpath)
            elif prob < 0.12:
                val.append(fpath)
            else:
                train.append(fpath)
    return train, val, test


def fit_pca(jpg_list):
    X = np.zeros((len(jpg_list), 201))
    for i, f in enumerate(jpg_list):
        arr = pd.read_csv(
            f.replace(".jpg", ".csv"), skiprows=range(5), names=["f", "v"]
        )["v"].values.astype(float)
        X[i, :] = arr
    pca = PCA(n_components=PCA_NCOMP).fit(X)
    # quick diagnostic bar-plot
    plt.figure(figsize=(6, 3))
    plt.bar(range(PCA_NCOMP), pca.explained_variance_ratio_)
    plt.step(range(PCA_NCOMP), np.cumsum(pca.explained_variance_ratio_), where="mid")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/pca_variance.png", dpi=200)
    plt.close()
    return pca


def preprocess_one(img_path, pca=None, augment=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    folder = os.path.basename(os.path.dirname(img_path))
    if augment and folder in ["0d65h", "0d75h"]:
        img = rotate_img(img, random.randint(0, 359))
    img = cv2.resize(img, (IMG_W, IMG_H)) / 255.0
    img = img.reshape(IMG_H, IMG_W, 1)

    prop = [int(x) for x in re.findall(r"\d+", folder)]
    prop = np.array([prop[0] / 60, prop[1] / 75])  # normalised
    p1, p2 = np.full((IMG_H, IMG_W, 1), prop[0]), np.full((IMG_H, IMG_W, 1), prop[1])
    img = np.concatenate([img, p1, p2], axis=-1)

    label = pd.read_csv(
        img_path.replace(".jpg", ".csv"), skiprows=range(5), names=["f", "v"]
    )["v"].values
    if pca is not None:
        label = pca.transform(label.reshape(1, -1)).ravel()
    return img.astype("float32"), label.astype("float32")


###############################################################################
# 3.   DATA LOADING
###############################################################################
train_files, val_files, test_files = collect_files()
print(f"files: train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")

pca = fit_pca(train_files)  # PCA fitted on training set ONLY


###############################################################################
# 4.   ViT – load reused weights (NO re-training)
###############################################################################
def build_vit_model(image_size, num_heads=4, num_transformer_layers=4):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    patch_size = PATCH_SIZE
    num_patches = (image_size // patch_size) ** 2
    patch_dim = 64
    patches = layers.Conv2D(
        filters=patch_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
    )(inputs)
    patches = layers.Reshape((num_patches, patch_dim))(patches)
    pos_enc = layers.Embedding(input_dim=num_patches, output_dim=patch_dim)(
        tf.range(start=0, limit=num_patches, delta=1)
    )
    patches = patches + pos_enc
    for _ in range(num_transformer_layers):
        x = layers.LayerNormalization(epsilon=1e-6)(patches)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=patch_dim)(
            x, x
        )
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(128, activation="relu")(x)
        ffn = layers.Dense(patch_dim)(ffn)
        patches = layers.Add()([x, ffn])
    aggregated = layers.GlobalAveragePooling1D()(patches)
    output = layers.Dense(PCA_NCOMP, activation="linear")(aggregated)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


vit_model = build_vit_model(IMG_H, num_heads=4, num_transformer_layers=4)
vit_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4), loss=weighted_mse, metrics=["mae"]
)
vit_model.load_weights("iteration_5_weights.h5")
vit_model.summary()


def evaluate_model(model, files, name):
    ys, preds = [], []
    for f in files:
        x, y = preprocess_one(f, pca=pca, augment=False)
        pred = model.predict(x[None, ...], verbose=1)[0]
        y_full, pred_full = (
            pca.inverse_transform(y[None, ...])[0],
            pca.inverse_transform(pred[None, ...])[0],
        )
        ys.append(y_full)
        preds.append(pred_full)
    mse = np.mean([mean_squared_error(y_, p_) for y_, p_ in zip(ys, preds)])
    print(f"{name}  MSE={mse:.5f}")
    return mse, ys, preds


vit_mse, vit_y, vit_pred = evaluate_model(vit_model, test_files, "ViT")


###############################################################################
# 5.   BASELINES – CNN & MLP  (trained from scratch)
###############################################################################
def build_cnn():
    i = layers.Input(shape=(IMG_H, IMG_W, 3))
    x = layers.Conv2D(16, 3, activation="relu")(i)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    o = layers.Dense(PCA_NCOMP)(x)
    return keras.Model(i, o)


def build_mlp():
    i = layers.Input(shape=(IMG_H, IMG_W, 3))
    x = layers.Flatten()(i)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    o = layers.Dense(PCA_NCOMP)(x)
    return keras.Model(i, o)


def make_dataset(file_list, shuffle=True):
    def gen():
        for f in file_list:
            x, y = preprocess_one(f, pca=pca)
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((IMG_H, IMG_W, 3), tf.float32),
            tf.TensorSpec((PCA_NCOMP,), tf.float32),
        ),
    )
    if shuffle:
        print(f"Shuffling {len(file_list)} samples (this may take a moment)…")
        ds = ds.shuffle(len(file_list)).repeat()
        # after you call shuffle().repeat(), you can immediately prefetch:
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("Dataset ready—entering training loop.")
    return ds


train_ds = make_dataset(train_files)
val_ds = make_dataset(val_files, shuffle=False).repeat()
test_ds = make_dataset(test_files, shuffle=False)


def train_baseline(build_fn, name):
    print(f"\n=== Starting {name} training ===")
    model = build_fn()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss=weighted_mse, metrics=["mae"]
    )
    cbs = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

    def _on_epoch_end(epoch, logs):
        loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            print(
                f"[{name}] Epoch {epoch+1} end: loss={loss:.4f}, val_loss={val_loss:.4f}"
            )
        else:
            print(f"[{name}] Epoch {epoch+1} end: loss={loss:.4f}")

    epoch_logger = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: print(f"[{name}] Epoch {epoch+1} start"),
        on_epoch_end=_on_epoch_end,
    )
    all_cbs = cbs + [epoch_logger]
    steps_per_epoch = len(train_files) // BATCH_SIZE
    val_steps = len(val_files) // BATCH_SIZE

    h = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=all_cbs,
        verbose=1,
    )

    # Save history safely even if arrays differ in length
    hist = h.history
    # Convert each metric list to a pandas Series so shorter lists are padded with NaN
    df_hist = pd.DataFrame({k: pd.Series(v) for k, v in hist.items()})
    df_hist.to_csv(f"{RESULT_DIR}/{name}_learning_curve.csv", index=False)

    mse, ys, preds = evaluate_model(model, test_files, name)
    model.save(f"{RESULT_DIR}/{name}.h5")
    # plot learning curve
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.title(f"{name} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{name}_curve.png", dpi=200)
    plt.close()
    return mse, ys, preds


cnn_mse, cnn_y, cnn_pred = train_baseline(build_cnn, "CNN")
mlp_mse, mlp_y, mlp_pred = train_baseline(build_mlp, "MLP")


###############################################################################
# 6.   RESULTS TABLE  (for paper)
###############################################################################
tbl = pd.DataFrame(
    {"Model": ["ViT", "CNN", "MLP"], "Test_MSE": [vit_mse, cnn_mse, mlp_mse]}
)
tbl.to_csv(f"{RESULT_DIR}/model_comparison.csv", index=False)
print(tbl)


###############################################################################
# 7.   SPECTRA OVERLAYS  (first 4 test samples for each model)
###############################################################################
def plot_spec(y_true, y_pred, tag, idx):
    plt.figure(figsize=(4, 2))
    plt.plot(y_true, label="FEM")
    plt.plot(y_pred, label=tag, ls="--")
    plt.title(f"Sample {idx} – {tag}")
    plt.ylabel("Absorption")
    plt.xlabel("Freq idx")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{tag}_spec_{idx}.png", dpi=200)
    plt.close()


for i in range(min(4, len(test_files))):
    plot_spec(vit_y[i], vit_pred[i], "ViT", i)
    plot_spec(cnn_y[i], cnn_pred[i], "CNN", i)
    plot_spec(mlp_y[i], mlp_pred[i], "MLP", i)


###############################################################################
# 8.   DATASET STATISTICS  (bins + augmentation count)
###############################################################################
def get_props(f):
    folder = os.path.basename(os.path.dirname(f))
    h, a = [int(x) for x in re.findall(r"\d+", folder)]
    return h, a


stats = pd.DataFrame(
    [get_props(f) for f in train_files + val_files + test_files],
    columns=["height_mm", "angle_deg"],
)
stats["split"] = (
    ["train"] * len(train_files) + ["val"] * len(val_files) + ["test"] * len(test_files)
)
stats.to_csv(f"{RESULT_DIR}/dataset_stats.csv", index=False)

print("\nAll artifacts are in:", RESULT_DIR)
