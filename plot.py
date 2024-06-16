import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams


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


# Load the model and weights
model = build_vit_model()
model.load_weights("iteration_5_weights.h5")

# Checking the layers to find one with weights
print("Model Layers:")
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# Choose an appropriate layer that has weights
layer_index = 32  # Adjust based on your actual layer's position
layer = model.layers[layer_index]

# Verify if the layer has weights
if layer.weights:
    weights = layer.get_weights()[0]

    # Setting matplotlib parameters for the font
    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = 10
    rcParams["figure.figsize"] = (8, 6)

    # Assuming 'weights' is your numpy array from the model's layer
    ax = sns.heatmap(weights, cmap="viridis", annot=False)
    plt.title("Heatmap of Weights in Layer: dense_8")
    plt.xlabel("Features")
    plt.ylabel("Weights")

    # Enhancing color bar
    cbar = ax.collections[0].colorbar
    cbar.set_label("Weight Magnitude")
    cbar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    cbar.set_ticklabels(["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"])

    plt.show()
else:
    print(f"No weights in layer: {layer.name}")
