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
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.common.ml_utils import (
    set_global_seed,
    compute_class_weights,
    plot_training_history,
    evaluate_keras_model_on_directory,
)


# ---------- Config ----------

IMG_SIZE = 224
NUM_CLASSES = 2  # 2 for Midjourney / DALL-E
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SEED = 42
TRAINABLE_BASE_LAYERS = 5  # how many base layers to unfreeze from the end

DATASET_ROOT = Path("dataset/dataset_224x224")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TESTSET_ROOT = Path("dataset/testset_224x224")
MODEL_SAVE_PATH = "midjourney_vs_dalle_efficientnet_detector.keras"

# Set random seeds for reproducibility
set_global_seed(SEED)


# ---------- Dataset ----------

def prepare_data(train_dir, val_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Prepare data generators for training and validation using ImageDataGenerator.

    Directory structure is expected as:
      train/
        dalle/
        midjourney/
      val/
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


# ---------- Model: EfficientNetB0 backbone + classifier ----------

def build_efficientnet_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, trainable_base_layers=0, lr=LR, label_smoothing=0.0):
    """
    Build transfer learning model using EfficientNetB0 as backbone.

    Args:
        img_size: Input image size (img_size x img_size).
        num_classes: Number of output classes.
        trainable_base_layers: Number of base model layers to make trainable (0 = freeze all).
        lr: Learning rate for the optimizer.
        label_smoothing: Label smoothing factor for loss function.
    """
    # Load pre-trained EfficientNetB0 model without top layers
    base_model = EfficientNetB0(
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

    # Preprocessing for EfficientNetB0 (scales to [-1, 1])
    x = preprocess_input(inputs)

    # Base model
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        metrics = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        metrics = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]

    model = keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )

    print("EfficientNetB0 model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")

    return model


# ---------- Training / Evaluation loops ----------

def train_model(model, train_generator, val_generator, epochs=EPOCHS):
    """
    Train the EfficientNetB0 model with callbacks and class weights.
    """
    if model is None:
        raise ValueError("Model not built. Call build_efficientnet_model() first.")

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

    # Class weights to handle class imbalance (supports binary and multi-class)
    class_weights = compute_class_weights(train_generator.labels)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return history


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
    plt.savefig("efficientnet_confusion_matrix.png", dpi=150)
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
    plt.savefig("dataset_comparison_efficientnet.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nComparison saved to 'dataset_comparison_efficientnet.png'")


def run_pipeline(
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    trainable_base_layers: int = TRAINABLE_BASE_LAYERS,
    dataset_root: Path = DATASET_ROOT,
    model_save_path: str = MODEL_SAVE_PATH,
    label_smoothing: float = 0.0,
    seed: int = SEED,
):
    """Run the full EfficientNetB0 pipeline end-to-end.

    All arguments have sensible defaults taken from the config section,
    but can be overridden when calling this function.
    """
    # Update seeds (in case caller overrides them)
    set_global_seed(seed)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    print(f"Using EfficientNetB0 with image size {img_size}x{img_size}")
    print(f"Dataset root: {dataset_root}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Trainable base layers: {trainable_base_layers}")

    # Basic sanity check on dataset paths and visualize some samples
    sanity_check_and_visualize(train_dir)

    # Build model
    model = build_efficientnet_model(
        img_size=img_size,
        num_classes=num_classes,
        trainable_base_layers=trainable_base_layers,
        lr=lr,
        label_smoothing=label_smoothing,
    )

    # Prepare data
    train_gen, val_gen = prepare_data(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        img_size=img_size,
    )

    # Train
    print("\nStarting training...")
    history = train_model(model, train_gen, val_gen, epochs=epochs)

    # Plot training history
    plot_training_history(history, title_prefix="EFFICIENTNET")

    # Evaluate on validation set
    print("\nEvaluating EFFICIENTNET model on VALIDATION set...")
    evaluate_model(model, val_gen)

    # --- Evaluate on TEST set (if exists) ---
    if TESTSET_ROOT.exists():
        print("\nEvaluating EFFICIENTNET model on TEST set...")
        evaluate_keras_model_on_directory(
            model=model,
            data_dir=TESTSET_ROOT,
            img_size=img_size,
            batch_size=batch_size,
            num_classes=num_classes,
            class_mode="binary" if num_classes == 2 else "categorical",
            save_confusion_matrix_path="efficientnet_confusion_matrix_test.png",
            title_prefix="EFFICIENTNET TEST",
        )
    else:
        print(f"\nNo TEST set found at: {TESTSET_ROOT}")

    # Save model (best model is already saved by ModelCheckpoint, but save final as well if desired)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, (train_gen, val_gen)


# ---------- Main ----------

def main():
    model, (train_gen, val_gen) = run_pipeline()
    return model, (train_gen, val_gen)


if __name__ == "__main__":
    main()
