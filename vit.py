# %%
# ## 0.  Install & Imports
# (Uncomment if running fresh environment)
# !pip install tensorflow==2.19.0 scikit-learn opencv-python matplotlib pandas

import os, re, glob, cv2, gc, sys, contextlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["GLOG_minloglevel"] = "3"  # Suppress GLOG messages
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC INFO and WARNING messages
os.environ["GRPC_CPP_LOG_LEVEL"] = "ERROR"  # Suppress gRPC C++ logs

import numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import silence_tensorflow.auto

import logging

logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf

tf.get_logger().setLevel("FATAL")

from tensorflow import keras
from keras import layers

# %%
# ## 1.  Configuration

# Paths
DATA_DIR = Path("data")  # expects subfolders of .jpg/.csv
OUTPUT_DIR = Path("outputs/notebook")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Image & PCA
IMG_H = IMG_W = 64
PCA_NCOMP = 40

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
BASE_LR = 1e-4
PATIENCE = 15

# ViT architecture
PATCH_SIZE = 16
VIT_NUM_HEADS = 3
VIT_LAYERS = 3
MLP_DIM = 64


# Random seed for reproducibility
SEED = 42123
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------------------------------------------------
# Inference / Pretrained Model Configuration
# -----------------------------------------------------------------------------
USE_PRETRAINED = False  # train a fresh model, then evaluate
PRETRAINED_MAIN = Path("iteration_5.h5")  # single-file saved model
PRETRAINED_MODEL_ONLY = Path("iteration_5_model.h5")  # architecture-only
PRETRAINED_WEIGHTS = Path("iteration_5_weights.h5")  # separate weights file


# %%
# ## 2.  Helper Functions


def natural_sort(filelist):
    """Sort file paths in human order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", str(key))]
    return sorted(filelist, key=alphanum_key)


def collect_images():
    jpgs = list(DATA_DIR.glob("*/*.jpg"))
    return natural_sort(jpgs)


def split_files(jpg_list, val_frac=0.2, test_frac=0.3, seed=SEED):
    rng = np.random.default_rng(seed)
    # extract (h,a) class from folder name
    keys = []
    for p in jpg_list:
        h, a = [int(x) for x in re.findall(r"\d+", p.parent.name)]
        keys.append((h, a))
    keys = np.array(keys)
    idxs = np.arange(len(jpg_list))
    # group by unique key; then sample within groups proportionally so all conditions appear in each split
    train_idx, val_idx, test_idx = [], [], []
    for k in np.unique(keys, axis=0):
        mask = (keys == k).all(axis=1)
        group = idxs[mask]
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test_idx.extend(group[:n_test])
        val_idx.extend(group[n_test : n_test + n_val])
        train_idx.extend(group[n_test + n_val :])
    return (
        [jpg_list[i] for i in train_idx],
        [jpg_list[i] for i in val_idx],
        [jpg_list[i] for i in test_idx],
    )


def load_spectrum(path):
    """Read .csv, skip first 5 lines, return float32 spectrum."""
    arr = pd.read_csv(path, skiprows=range(5), names=["f", "v"])["v"].to_numpy(
        dtype="float32"
    )
    return arr


def fit_pca(train_paths):
    # gather only spectra of consistent length
    spectra_list = []
    expected_len = None
    for p in train_paths:
        arr = load_spectrum(p.with_suffix(".csv"))
        if expected_len is None:
            expected_len = arr.shape[0]
        if arr.shape[0] != expected_len:
            print(f"Skipping {p.name}: length {arr.shape[0]} (expected {expected_len})")
            continue
        spectra_list.append(arr)
    if not spectra_list:
        raise ValueError(
            "No valid spectra found for PCA. Check CSV files for row consistency."
        )
    spectra = np.stack(spectra_list)
    pca = PCA(n_components=PCA_NCOMP, random_state=SEED).fit(spectra)
    return pca


def parse_one(path, pca, augment=True):
    """Load JPG + CSV, apply optional rotation, pack props into channels."""
    # load grayscale image
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    folder = path.parent.name

    # augment: random rotation for specific folders
    if augment and folder in {"0d65h", "0d75h"}:
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REFLECT)
    else:
        img = cv2.resize(img, (IMG_W, IMG_H))

    # normalize
    img = img.astype("float32") / 255.0
    img = img[..., None]  # shape (H,W,1)

    # extract numeric properties from folder name, map to [0,1]
    h, a = [int(x) for x in re.findall(r"\d+", folder)]
    prop = np.array([h / 60.0, a / 75.0], dtype="float32")
    prop1 = np.full((IMG_H, IMG_W, 1), prop[0], dtype="float32")
    prop2 = np.full((IMG_H, IMG_W, 1), prop[1], dtype="float32")
    img = np.concatenate([img, prop1, prop2], axis=-1)  # shape (H,W,3)

    # PCA-transform the spectrum
    spectrum = load_spectrum(path.with_suffix(".csv"))
    y = pca.transform(spectrum.reshape(1, -1))[0].astype("float32")
    return img, y


def make_tf_dataset(paths, pca, training=False):
    """Build a tf.data.Dataset for a list of Paths."""
    AUTOTUNE = tf.data.AUTOTUNE

    def gen():
        expected_dim = pca.components_.shape[1]
        for p in paths:
            spec = load_spectrum(p.with_suffix(".csv"))
            if spec.shape[0] != expected_dim:
                print(
                    f"Skipping {p.name}: spectrum length {spec.shape[0]} ≠ expected {expected_dim}"
                )
                continue
            yield parse_one(p, pca, augment=training)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((IMG_H, IMG_W, 3), tf.float32),
            tf.TensorSpec((PCA_NCOMP,), tf.float32),
        ),
    )
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def weighted_mse(y_true, y_pred):
    idx = tf.cast(tf.range(1, PCA_NCOMP + 1), tf.float32)
    return tf.reduce_mean((1.0 / idx) * tf.square(y_true - y_pred))


# %%
# ## 3.  Build the Vision Transformer


def patch_embed(x):
    x = layers.Conv2D(
        filters=MLP_DIM,
        kernel_size=(PATCH_SIZE, PATCH_SIZE),
        strides=(PATCH_SIZE, PATCH_SIZE),
        padding="valid",
    )(x)
    num_patches = (IMG_H // PATCH_SIZE) ** 2
    x = layers.Reshape((num_patches, MLP_DIM))(x)
    return x


def build_vit():
    inp = layers.Input((IMG_H, IMG_W, 3))
    x = patch_embed(inp)

    # learnable positional embedding
    pos = layers.Embedding(input_dim=x.shape[1], output_dim=MLP_DIM)(
        tf.range(start=0, limit=x.shape[1], delta=1)
    )
    x = x + pos

    # transformer encoder blocks
    for _ in range(VIT_LAYERS):
        # attention block
        y = layers.LayerNormalization(epsilon=1e-6)(x)
        y = layers.MultiHeadAttention(num_heads=VIT_NUM_HEADS, key_dim=MLP_DIM)(y, y)
        x = layers.Add()([x, y])

        # MLP block
        y = layers.LayerNormalization(epsilon=1e-6)(x)
        y = layers.Dense(MLP_DIM * 4, activation="gelu")(y)
        y = layers.Dense(MLP_DIM)(y)
        x = layers.Add()([x, y])

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(PCA_NCOMP, activation="linear")(x)
    return keras.Model(inputs=inp, outputs=out, name="ViT_PCA")


# %%
# ## 4.  Prepare Data & PCA

all_jpg = collect_images()
train_files, val_files, test_files = split_files(all_jpg)
print(f"Splits: train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")

pca = fit_pca(train_files)
print("PCA fitted, explained variance sum:", pca.explained_variance_ratio_.sum())

train_ds = make_tf_dataset(train_files, pca, training=True)
val_ds = make_tf_dataset(val_files, pca, training=False).repeat()
test_ds = make_tf_dataset(test_files, pca, training=False)


# %%
# ## 5.  Compile & Train

if not USE_PRETRAINED:
    strategy = tf.distribute.get_strategy()  # change if multi-GPU
    with strategy.scope():
        model = build_vit()
        opt = tf.keras.optimizers.Adam(BASE_LR)
        model.compile(optimizer=opt, loss=weighted_mse, metrics=["mae"])

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=PATIENCE // 2, factor=0.5),
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "vit_best.keras"), save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.TensorBoard(str(OUTPUT_DIR / "tb")),
    ]
    steps_per_epoch = len(train_files) // BATCH_SIZE
    val_steps = len(val_files) // BATCH_SIZE

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "history.csv", index=False)
    model.save(OUTPUT_DIR / "vit_final.keras")
else:
    print("Loading pretrained model...")
    model = None
    if PRETRAINED_MAIN.exists():
        try:
            # Try to load a full saved model (architecture + weights)
            model = keras.models.load_model(
                PRETRAINED_MAIN,
                custom_objects={"weighted_mse": weighted_mse},
                compile=False,  # we'll compile after successful load
            )
            print(f"Loaded full model from {PRETRAINED_MAIN}.")
        except Exception as e:
            print(f"⚠️ Could not load full model from {PRETRAINED_MAIN}: {e}")
            model = None  # force fallback
    # Fallback path: build architecture & load separate weights
    if model is None:
        if PRETRAINED_MODEL_ONLY.exists():
            try:
                model = keras.models.load_model(
                    PRETRAINED_MODEL_ONLY,
                    custom_objects={"weighted_mse": weighted_mse},
                    compile=False,
                )
                print(f"Loaded architecture from {PRETRAINED_MODEL_ONLY}.")
            except Exception as e:
                print(
                    f"⚠️ Could not load architecture from {PRETRAINED_MODEL_ONLY}: {e}"
                )
                model = None
        if model is None:
            # Final fallback: rebuild architecture from code
            print("Rebuilding architecture via build_vit() fallback.")
            model = build_vit()
        # Load weights if available
        if PRETRAINED_WEIGHTS.exists():
            try:
                model.load_weights(PRETRAINED_WEIGHTS)
                print(f"Loaded weights from {PRETRAINED_WEIGHTS}.")
            except Exception as e:
                print(f"⚠️ Failed to load weights from {PRETRAINED_WEIGHTS}: {e}")
        else:
            print(
                f"⚠️ Weights file {PRETRAINED_WEIGHTS} not found; using random weights."
            )
    # compile for inference/evaluation
    model.compile(
        optimizer=keras.optimizers.Adam(BASE_LR), loss=weighted_mse, metrics=["mae"]
    )
    model.summary()

# %%

# ## 6.  Evaluate on Test Set

# Directory to save evaluation plots
# Instead of reusing/overwriting the same folder every run, auto-increment an
# iteration directory (iteration_001, iteration_002, ...) under plot_output.
PLOT_ROOT = Path("plot_output")


def _next_iteration_dir(root=PLOT_ROOT):
    """
    Create and return the next iteration directory under `root`.

    Scans for subdirectories named `iteration_<int>` (zero-padded not required),
    finds the largest existing integer, and returns a newly created directory
    with that integer + 1. Returns (path, int).
    """
    root.mkdir(exist_ok=True, parents=True)
    max_n = 0
    for p in root.glob("iteration_*"):
        if p.is_dir():
            m = re.match(r"iteration_(\d+)$", p.name)
            if m:
                try:
                    n = int(m.group(1))
                except ValueError:
                    continue
                if n > max_n:
                    max_n = n
    next_n = max_n + 1
    iter_dir = root / f"iteration_{next_n:03d}"  # zero-pad for neat sorting
    iter_dir.mkdir(exist_ok=False, parents=True)
    return iter_dir, next_n


PLOT_DIR, ITERATION_NUM = _next_iteration_dir()
print(f"Saving evaluation plots to {PLOT_DIR} (iteration {ITERATION_NUM}).")

mse_list = []
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch, verbose=1)
    mses = np.mean((y_batch - preds) ** 2, axis=1)
    mse_list.extend(mses.tolist())

print("Mean Test MSE:", np.mean(mse_list))

# plot a few spectra overlays
import matplotlib.pyplot as plt

count = 0
test_files_iter = iter(test_files)  # Create iterator to track filenames
for x_batch, y_batch in test_ds.take(1):
    preds = model.predict(x_batch, verbose=1)
    inv_true = pca.inverse_transform(y_batch)
    inv_pred = pca.inverse_transform(preds)
    for i in range(min(21, len(inv_true))):
        # Get the corresponding filename
        try:
            current_file = next(test_files_iter)
            filename = current_file.name
        except StopIteration:
            filename = f"test_sample_{count}"

        plt.figure()
        plt.plot(inv_true[i], label="True")
        plt.plot(inv_pred[i], "--", label="Pred")
        plt.legend()
        plt.title(f"Sample {count}")
        plt.ylim(-1.0, 1.0)
        # plt.yticks(np.linspace(-1.0, 1.0, 9))
        # Add filename beneath the plot
        plt.figtext(
            0.5, 0.02, f"File: {filename}", ha="center", fontsize=10, style="italic"
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for filename
        # Save plot instead of (or in addition to) showing interactively
        out_path = PLOT_DIR / f"{Path(filename).stem}_overlay.png"
        plt.savefig(out_path, dpi=150)
        plt.close()  # free memory; comment out if you prefer interactive display
        # plt.show()  # uncomment to display inline
        count += 1

# %%
print("Notebook run complete! All artefacts in", OUTPUT_DIR)


# # %% Cell 7: 5-FOLD CROSS-VALIDATION (C-2), with per-sample validation
# from sklearn.model_selection import KFold
# import json, time


# def kfold_vit(build_fn, all_files, k=5, epochs=EPOCHS):
#     kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
#     mses, times = [], []
#     for fold, (tr_idx, vl_idx) in enumerate(kf.split(all_files), 1):
#         print(f"\n▶ Fold {fold}/{k}")
#         tr = [all_files[i] for i in tr_idx]
#         vl = [all_files[i] for i in vl_idx]

#         ds_tr = make_tf_dataset(tr, pca, training=True)
#         ds_vl = make_tf_dataset(vl, pca, training=False)

#         # build & train
#         m = build_fn()
#         m.compile(optimizer=keras.optimizers.Adam(BASE_LR), loss=weighted_mse)
#         steps_per_epoch = len(train_files) // BATCH_SIZE
#         val_steps = len(val_files) // BATCH_SIZE
#         m.fit(
#             ds_tr,
#             epochs=epochs,
#             callbacks=[
#                 keras.callbacks.EarlyStopping(
#                     patience=PATIENCE, restore_best_weights=True
#                 )
#             ],
#             steps_per_epoch=steps_per_epoch,
#             validation_steps=val_steps,
#             verbose=1,
#         )

#         # measure inference-time
#         t0 = time.perf_counter()
#         _ = m.predict(ds_vl, verbose=0)
#         t1 = time.perf_counter()
#         sec_sample = (t1 - t0) / len(vl)

#         # evaluate MSE on each valid sample
#         ys, preds = [], []
#         for p in vl:
#             try:
#                 x, y = parse_one(p, pca, augment=False)
#             except ValueError as e:
#                 print(f"  Skipping {p.name}: {e}")
#                 continue
#             pr = m.predict(x[None, ...], verbose=0)[0]
#             ys.append(pca.inverse_transform(y[None, :])[0])
#             preds.append(pca.inverse_transform(pr[None, :])[0])

#         if not ys:
#             print(f"  No valid samples in fold {fold}, skipping metrics.")
#             continue

#         fold_mse = np.mean([mean_squared_error(a, b) for a, b in zip(ys, preds)])
#         print(f"  Fold {fold} MSE={fold_mse:.5f}, {sec_sample*1000:.2f} ms/sample")
#         mses.append(fold_mse)
#         times.append(sec_sample)

#         tf.keras.backend.clear_session()
#         gc.collect()

#     summary = {
#         "mse_mean": float(np.mean(mses)),
#         "mse_std": float(np.std(mses)),
#         "avg_ms_per_sample": float(np.mean(times)) * 1000,
#     }
#     with open(OUTPUT_DIR / "vit_kfold.json", "w") as f:
#         json.dump(summary, f, indent=2)
#     print("✅ K-fold summary written to", OUTPUT_DIR / "vit_kfold.json", "\n", summary)


# # run it
# kfold_vit(build_vit, train_files, k=5)


# # %% Cell 8: OUT-OF-DISTRIBUTION EVALUATION (C-3), skipping bad files
# def is_ood(path, h_min=68, h_max=72, max_angle=50):
#     h, a = [int(x) for x in re.findall(r"\d+", path.parent.name)]
#     return (h < h_min or h > h_max) or (a >= max_angle)


# ood_files = [p for p in train_files + val_files + test_files if is_ood(p)]
# print(f"OOD candidate set size: {len(ood_files)}")

# ys, preds = [], []
# for p in ood_files:
#     try:
#         x, y = parse_one(p, pca, augment=False)
#     except ValueError as e:
#         print(f"  Skipping {p.name}: {e}")
#         continue
#     pr = model.predict(x[None, ...], verbose=0)[0]
#     ys.append(pca.inverse_transform(y[None, :])[0])
#     preds.append(pca.inverse_transform(pr[None, :])[0])

# if ys:
#     ood_mse = np.mean([mean_squared_error(a, b) for a, b in zip(ys, preds)])
#     print(f"✅ ViT OOD MSE = {ood_mse:.5f}")
#     pd.DataFrame(
#         {
#             "file": [p.name for p in ood_files[: len(ys)]],
#             "mse": [mean_squared_error(a, b) for a, b in zip(ys, preds)],
#         }
#     ).to_csv(OUTPUT_DIR / "vit_ood_details.csv", index=False)
# else:
#     print("⚠️  No valid OOD samples found.")


# # %% Inverse-Design (improved)
# import tensorflow as tf


# def inverse_design_constrained(target_csv, steps=300, lr=1e-1, tv_weight=1e-2):
#     # 1. load & PCA-project target spectrum
#     spec = load_spectrum(Path(target_csv))
#     if spec.shape[0] != pca.n_features_in_:
#         raise ValueError("CSV length mismatch")
#     t_pca = pca.transform(spec.reshape(1, -1))[0]

#     # 2. build initial guess: start from a uniform gray geometry
#     #    and freeze channels 1 & 2 to the known physical props
#     #    parse_one() can give us prop-channels for one sample:
#     example_img, _ = parse_one(train_files[0], pca, augment=False)
#     prop1 = example_img[..., 1:2]  # channel 1
#     prop2 = example_img[..., 2:3]  # channel 2

#     # trainable geometry channel (1): initialize to 0.5
#     geom = tf.Variable(tf.ones([1, IMG_H, IMG_W, 1]) * 0.5, dtype=tf.float32)

#     opt = tf.keras.optimizers.Adam(lr)
#     best_l, best_geom = 1e9, None

#     for step in range(1, steps + 1):
#         with tf.GradientTape() as tape:
#             # assemble full input: [geom, prop1, prop2]
#             x = tf.concat([geom, prop1[None, ...], prop2[None, ...]], axis=-1)
#             pred = model(x, training=False)  # ViT forward
#             err = tf.reduce_mean((pred - t_pca) ** 2)  # spectrum MSE

#             # total variation loss on geometry for smoothness
#             tv = tf.image.total_variation(geom)[0]
#             loss = err + tv_weight * tv

#         grads = tape.gradient(loss, [geom])
#         opt.apply_gradients(zip(grads, [geom]))
#         # clamp geometry to [0,1]
#         geom.assign(tf.clip_by_value(geom, 0.0, 1.0))

#         # track best
#         if loss < best_l:
#             best_l, best_geom = float(loss), geom.numpy().copy()

#         if step % 50 == 0:
#             print(f"step {step:>3d}  err={err:.4e}  tv={tv:.4e}")

#     # 3. threshold geometry to binary mask
#     bin_geom = (best_geom[0, ..., 0] > 0.5).astype(np.uint8) * 255
#     out_png = OUTPUT_DIR / f"inv_design_bin_{Path(target_csv).stem}.png"
#     cv2.imwrite(str(out_png), bin_geom)

#     # 4. export the predicted spectrum
#     full_input = np.stack(
#         [bin_geom / 255, prop1[..., :, 0], prop2[..., :, 0]], axis=-1
#     )[None, ...]
#     pred_pca = model(full_input, training=False)[0].numpy()
#     pred_spec = pca.inverse_transform(pred_pca)
#     pd.DataFrame({"target": spec, "predicted": pred_spec}).to_csv(
#         OUTPUT_DIR / f"inv_design_{Path(target_csv).stem}_spec.csv", index=False
#     )

#     print(f"Saved binary design → {out_png}  (best loss {best_l:.4e})")
#     return out_png


# Example usage:
# inverse_design_constrained("data/0d65h/sample_123.csv")
