import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------- Config ----------

IMG_SIZE = 299
NUM_CLASSES = 2  # 2 for Midjourney / DALL-E
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SEED = 42
TRAINABLE_BASE_LAYERS = 5  # how many base layers to unfreeze from the end

DATASET_ROOT = Path("dataset/dataset_299x299")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
MODEL_SAVE_PATH = "midjourney_vs_dalle_xception_detector.keras"

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ---------- Dataset ----------

def prepare_data(train_dir, val_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Prepare data generators for training and validation using ImageDataGenerator.

    Directory structure is expected as:
      train_dir/
        dalle/
        midjourney/
      val_dir/
        dalle/
        midjourney/
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator()

    class_mode = "binary" if NUM_CLASSES == 2 else "categorical"

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True,
        seed=SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False,
    )

    print("\nData prepared:")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")

    return train_generator, val_generator


# ---------- Model: Xception backbone + classifier ----------

def build_xception_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, trainable_base_layers=0):
    """
    Build transfer learning model using Xception as backbone.

    Args:
        img_size: Input image size (img_size x img_size).
        num_classes: Number of output classes.
        trainable_base_layers: Number of base model layers to make trainable (0 = freeze all).
    """
    # Load pre-trained Xception model without top layers
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    # Freeze base model layers by default
    base_model.trainable = False

    # If specified, make the last N layers trainable for fine-tuning
    if trainable_base_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_base_layers]:
            layer.trainable = False

    # Build the model
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Preprocessing for Xception (scales to [-1, 1])
    x = preprocess_input(inputs)

    # Base model
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "categorical_crossentropy"
        metrics = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]

    model = keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=loss,
        metrics=metrics,
    )

    print("Xception model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")

    return model


# ---------- Training / Evaluation loops ----------

def train_model(model, train_generator, val_generator, epochs=EPOCHS):
    """
    Train the Xception model with callbacks and class weights.
    """
    if model is None:
        raise ValueError("Model not built. Call build_xception_model() first.")

    # Default callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    # Class weights for binary case
    if NUM_CLASSES == 2:
        labels = np.array(train_generator.labels)
        class_weights = {
            0: len(labels) / (2 * np.sum(labels == 0)),
            1: len(labels) / (2 * np.sum(labels == 1)),
        }
    else:
        class_weights = None

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return history


def plot_training_history(history):
    """
    Plot training and validation accuracy and loss curves.
    """
    if history is None:
        print("No training history available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_generator):
    """
    Evaluate the model and show confusion matrix + classification report.
    """
    # Get predictions
    predictions = model.predict(val_generator)

    if NUM_CLASSES == 2:
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(predictions, axis=1)

    y_true = val_generator.classes

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(val_generator.class_indices.keys()),
        yticklabels=list(val_generator.class_indices.keys()),
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("xception_confusion_matrix.png", dpi=150)
    plt.show()


# ---------- Helper: dataset sanity + visualization ----------

def sanity_check_and_visualize(train_dir):
    """
    Sanity check the dataset structure and visualize a few DALL-E / Midjourney samples.
    """
    dalle_dir = train_dir / "dalle"
    mid_dir = train_dir / "midjourney"

    dalle_files = list(dalle_dir.glob("*"))
    mid_files = list(mid_dir.glob("*"))

    print(f"DALL-E samples: {len(dalle_files)}")
    print(f"Midjourney samples: {len(mid_files)}")

    if len(dalle_files) == 0 or len(mid_files) == 0:
        raise RuntimeError(
            f"No samples found in {train_dir}. "
            f"Please run data preparation and verify the dataset structure."
        )

    # Display a few from each for visual sanity check
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle("DALL-E vs Midjourney Comparison", fontsize=16)

    for i in range(3):
        img1 = Image.open(random.choice(dalle_files))
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"DALL-E Sample {i + 1}")
        axes[i, 0].axis("off")

        img2 = Image.open(random.choice(mid_files))
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"Midjourney Sample {i + 1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_comparison_xception.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nComparison saved to 'dataset_comparison_xception.png'")


# ---------- Main ----------

def main():
    print(f"Using Xception with image size {IMG_SIZE}x{IMG_SIZE}")
    print(f"Dataset root: {DATASET_ROOT}")

    # Basic sanity check on dataset paths and visualize some samples
    sanity_check_and_visualize(TRAIN_DIR)

    # Build model
    model = build_xception_model(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        trainable_base_layers=TRAINABLE_BASE_LAYERS,
    )

    # Prepare data
    train_gen, val_gen = prepare_data(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )

    # Train
    print("\nStarting training...")
    history = train_model(model, train_gen, val_gen, epochs=EPOCHS)

    # Plot training history
    plot_training_history(history)

    # Evaluate on validation set
    print("\nEvaluating XCEPTION model on VALIDATION set...")
    evaluate_model(model, val_gen)

    # Save model (best model is already saved by ModelCheckpoint, but save final as well if desired)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return model, (train_gen, val_gen)


if __name__ == "__main__":
    main()
