import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
import numpy as np
import cv2
import os
import glob
import re


def build_vit_model():
    input_shape = (64, 64, 3)
    inputs = layers.Input(shape=input_shape)

    # Convert image to patches
    patch_size = 8
    num_patches = (input_shape[0] // patch_size) ** 2
    patch_dim = 64
    patches = layers.Conv2D(
        filters=patch_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
    )(inputs)
    patches = layers.Reshape((num_patches, patch_dim))(patches)

    # Positional encoding
    pos_enc = layers.Embedding(input_dim=num_patches, output_dim=patch_dim)(
        tf.range(start=0, limit=num_patches, delta=1)
    )
    patches += pos_enc

    # Transformer layers
    for _ in range(4):
        x = layers.LayerNormalization(epsilon=1e-6)(patches)
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=patch_dim)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(128, activation="relu")(x)
        ffn = layers.Dense(patch_dim)(ffn)
        patches = layers.Add()([x, ffn])

    aggregated = layers.GlobalAveragePooling1D()(patches)
    outputs = layers.Dense(20, activation="linear")(aggregated)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_and_preprocess_images(file_path, image_size=(64, 64)):
    images = []
    for line in open(file_path, "r"):
        file_path = line.strip()
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        images.append(img)
    return np.array(images)


model = build_vit_model()
model.load_weights("iteration_5_weights.h5")
X_test = load_and_preprocess_images("test_files.txt")

# Using GradientTape for a simple gradient-based explanation
with tf.GradientTape() as tape:
    tape.watch(X_test)
    predictions = model(X_test)

grads = tape.gradient(predictions, X_test)

plt.figure(figsize=(12, 7))
for i in range(5):  # Adjust this to display the number of images you'd like
    plt.subplot(1, 5, i + 1)
    plt.imshow(grads[i], cmap="viridis")
    plt.axis("off")
plt.suptitle("Gradient-based feature importance")
plt.show()
