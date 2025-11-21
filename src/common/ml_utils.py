# Used to include different utility functions:

def set_global_seed(seed):
    '''
    Set all relevant random seeds (Python, NumPy, TensorFlow) to ensure reproducible
    training runs across pipelines.

    Args:
        seed (int): The seed value to apply across all supported libraries.

    Notes:
        - This function does *not* set PyTorch seeds. Those are initialized separately
          inside the CLIP pipeline (which uses PyTorch).
        - Calling this function helps ensure deterministic model initialization,
          shuffling, and preprocessing steps for TensorFlow‑based pipelines.
    '''

    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dataset_has_images(dataset_root):
    '''
    Validate that a dataset directory contains both `train/` and `val/` subfolders
    and that each of them contains at least one image file.

    Args:
        dataset_root (Path): Path to the dataset root directory.

    Raises:
        RuntimeError:
            - If either `train/` or `val/` does not exist under the given dataset root.
            - If either directory exists but contains no files (recursively).

    Purpose:
        This is used before running any training pipeline in order to avoid
        silent failures or confusing TensorFlow/PyTorch errors when attempting
        to load an empty or malformed dataset.
    '''
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            f"Missing 'train' or 'val' under {dataset_root}.\n"
            f"Expected:\n  {train_dir}\n  {val_dir}"
        )

    if not any(train_dir.rglob("*")):
        raise RuntimeError(f"No images found in: {train_dir}")
    if not any(val_dir.rglob("*")):
        raise RuntimeError(f"No images found in: {val_dir}")


def compute_class_weights(labels):
    '''
    Compute class weights for imbalanced classification problems from a sequence
    of integer labels. Supports both binary and multi-class scenarios.

    The weighting scheme is:

        w_c = N / (K * n_c)

    where:
        - N  is the total number of samples
        - K  is the number of unique classes
        - n_c is the number of samples belonging to class c

    This generalizes the common binary formula and can be passed directly to
    `class_weight` in `tf.keras.Model.fit`.

    Args:
        labels (Sequence[int]): Iterable of integer class labels (e.g. the
            `.labels` attribute from a Keras ImageDataGenerator or a simple
            NumPy / Python list of label indices).

    Returns:
        dict: A mapping from class index (int) to weight (float), e.g.:
              `{0: 0.7, 1: 1.3, 2: 2.1}`

    Notes:
        - This function assumes labels are integer-encoded (0, 1, 2, ...),
          which is the default for most Keras/TensorFlow image generators.
        - For binary classification, this reduces to the familiar
          `N / (2 * n_c)` expression for each class c.
    '''
    import numpy as np

    labels = np.array(labels)
    if labels.size == 0:
        raise ValueError("Cannot compute class weights: 'labels' is empty.")

    class_weights = {}
    unique_classes = np.unique(labels)
    total_samples = float(len(labels))
    num_classes = float(len(unique_classes))

    for c in unique_classes:
        count_c = float(np.sum(labels == c))
        if count_c == 0:
            # Should not happen given np.unique, but we guard anyway
            continue
        class_weights[int(c)] = total_samples / (num_classes * count_c)

    return class_weights

def plot_training_history(history, title_prefix: str = "Model"):
    """
    Plot training and validation accuracy and loss curves.

    Args:
        history: Keras History object returned by model.fit().
        title_prefix: String prefix to use in plot titles (e.g. 'Xception', 'EfficientNet').
    """

    import matplotlib.pyplot as plt

    if history is None:
        print("No training history available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title( f"{title_prefix} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{title_prefix} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{title_prefix.lower()}_training_curve.png", dpi=150)
    plt.show()

def plot_training_history_torch(train_losses, val_losses, train_accs, val_accs, title_prefix="CLIP"):
    """
    Plot training & validation loss + accuracy curves for PyTorch pipelines.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs, label="Val Acc")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{title_prefix.lower()}_training_curve.png", dpi=150)
    plt.show()