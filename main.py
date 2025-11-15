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
from tensorflow.keras.applications import EfficientNetB0, Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class AIImageDetector:
    def __init__(self, img_size=128, num_classes=2):
        """
        Initialize the AI image detector with transfer learning

        Args:
            img_size: Input image size (default 128x128)
            num_classes: Number of classes (2 for Midjourney/DALL-E, 3 for real/DALLE/Midjourney)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_modelX(self, trainable_base_layers=0):
        """
        Build transfer learning model using Xception

        Args:
            trainable_base_layers: Number of base model layers to make trainable (0 = freeze all)
        """
        # Load pre-trained Xception model without top layers
        base_model = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )

        # Freeze base model layers
        base_model.trainable = False

        # If specified, make the last N layers trainable for fine-tuning
        if trainable_base_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_base_layers]:
                layer.trainable = False

        # Build the model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))

        # Preprocessing for Xception (scales to [-1, 1])
        x = preprocess_input(inputs)

        # Base model
        x = base_model(x, training=False)

        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'

        self.model = keras.Model(inputs, outputs)

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print(f"Xception model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")

        return self.model

    def build_model(self, trainable_base_layers=0):
        """
        Build transfer learning model using EfficientNetB0

        Args:
            trainable_base_layers: Number of base model layers to make trainable (0 = freeze all)
        """
        # Load pre-trained EfficientNetB0 model without top layers
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )

        # Freeze base model layers
        base_model.trainable = False

        # If specified, make the last N layers trainable for fine-tuning
        if trainable_base_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_base_layers]:
                layer.trainable = False

        # Build the model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))

        # Data augmentation layers (applied during training only)
        # x = layers.RandomFlip("horizontal")(inputs)
        # x = layers.RandomRotation(0.1)(x)
        # x = layers.RandomZoom(0.1)(x)

        # Preprocessing for EfficientNet
        x = keras.applications.efficientnet.preprocess_input(inputs)

        # Base model
        x = base_model(x, training=False)

        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'

        self.model = keras.Model(inputs, outputs)

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print(f"Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")

        return self.model

    def prepare_data(self, train_dir, val_dir, batch_size=32):
        """
        Prepare data generators for training, validation, and optionally testing

        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            batch_size: Batch size for training

        Returns:
            train_generator, val_generator, (test_generator if test_dir provided)
        """
        # Note: We're using minimal augmentation here since we have augmentation in the model
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator()

        class_mode = 'binary' if self.num_classes == 2 else 'categorical'

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False
        )

        print(f"\nData prepared:")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")

        return train_generator, val_generator

    def train(self, train_generator, val_generator, epochs=20, callbacks=None):
        """
        Train the model

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs to train
            callbacks: List of Keras callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=7,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    '../best_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]

        class_weights = {
            0: len(train_generator.labels) / (2 * np.sum(train_generator.labels == 0)),
            1: len(train_generator.labels) / (2 * np.sum(train_generator.labels == 1))
        }

        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )

        return self.history

    def fine_tune(self, train_generator, val_generator, epochs=10, unfreeze_layers=50):
        """
        Fine-tune the model by unfreezing some base layers

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs for fine-tuning
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        # Unfreeze the base model layers
        base_model = self.model.layers[4]  # Get the EfficientNet base model
        base_model.trainable = True

        # Freeze all layers except the last unfreeze_layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        # Recompile with lower learning rate
        loss = 'binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy'
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print(f"\nFine-tuning with {unfreeze_layers} unfrozen layers...")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")

        # Continue training
        history_fine = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

        return history_fine

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate(self, test_generator):
        """
        Evaluate the model and show confusion matrix

        Args:
            test_generator: Test data generator
        """
        # Get predictions
        predictions = self.model.predict(test_generator)

        if self.num_classes == 2:
            y_pred = (predictions > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(predictions, axis=1)

        y_true = test_generator.classes

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(test_generator.class_indices.keys()),
                    yticklabels=list(test_generator.class_indices.keys()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def predict_image(self, image_path):
        """
        Predict whether a single image is AI-generated or real

        Args:
            image_path: Path to the image file

        Returns:
            prediction: Class prediction and probability
        """
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_size, self.img_size)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = self.model.predict(img_array, verbose=0)

        if self.num_classes == 2:
            prob = prediction[0][0]
            class_name = "Midjourney" if prob > 0.5 else "Dall-E"
            confidence = prob if prob > 0.5 else 1 - prob
        else:
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
            class_name = list(self.class_indices.keys())[class_idx]

        return class_name, confidence

    def save_model(self, filepath='ai_image_detector.keras'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='ai_image_detector.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Choose which model to run in this execution:
    #   "efficientnet"  -> EfficientNetB0 with 224x224 images
    #   "xception"      -> Xception with 299x299 images
    MODEL_TYPE = "efficientnet"  # change to "xception" when you want to run that one
    MODEL_TYPE = "xception"

    trainable_layers = 10

    if MODEL_TYPE == "efficientnet":
        img_size = 224
        dataset_root = Path("dataset/dataset_224x224")
        build_fn = "efficientnet"
        model_save_path = "midjourney_vs_dalle_efficientNet_detector.keras"
    elif MODEL_TYPE == "xception":
        img_size = 299
        dataset_root = Path("dataset/dataset_299x299")
        build_fn = "xception"
        model_save_path = "midjourney_vs_dalle_xception_detector.keras"
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    # Basic sanity check on dataset paths
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    dalle_dir = train_dir / "dalle"
    mid_dir = train_dir / "midjourney"

    dalle_files = list(dalle_dir.glob("*"))
    mid_files = list(mid_dir.glob("*"))

    print(f"Dalle samples: {len(dalle_files)}")
    print(f"Midjourney samples: {len(mid_files)}")

    if len(dalle_files) == 0 or len(mid_files) == 0:
        raise RuntimeError(
            f"No samples found in {train_dir}. "
            f"Please run data_prep.py and verify the dataset structure."
        )

    # Display a few from each for visual sanity check
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle('DALL-E vs Midjourney Comparison', fontsize=16)

    for i in range(3):
        img1 = Image.open(random.choice(dalle_files))
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f'DALL-E Sample {i + 1}')
        axes[i, 0].axis('off')

        img2 = Image.open(random.choice(mid_files))
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f'Midjourney Sample {i + 1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nComparison saved to 'dataset_comparison.png'")

    # Initialize detector with the chosen image size
    detector = AIImageDetector(img_size=img_size, num_classes=2)

    # Build the chosen model
    if build_fn == "efficientnet":
        detector.build_model(trainable_base_layers=trainable_layers)
    else:  # xception
        detector.build_modelX(trainable_base_layers=trainable_layers)

    # Show model architecture
    detector.model.summary()

    # Prepare data using the matching dataset resolution
    train_gen, val_gen = detector.prepare_data(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        batch_size=32
    )

    # Train the model
    print("\nStarting training...")
    history = detector.train(train_gen, val_gen, epochs=20)
    detector.plot_training_history()

    # Optional: Fine-tune for better performance
    # print("\nFine-tuning the model...")
    # history_fine = detector.fine_tune(train_gen, val_gen, epochs=10, unfreeze_layers=50)
    # Evaluate on validation set (serving as test here)

    print(f"\nEvaluating {MODEL_TYPE.upper()} model on VALIDATION set...")
    detector.evaluate(val_gen)

    # Save model
    detector.save_model(model_save_path)

    # Predict single image
    # detector = AIImageDetector(img_size=224, num_classes=2)
    # detector.load_model('midjourney_vs_dalle_detector.keras')
    # detector.prepare_data(
    #         train_dir='dataset_224x224/train',
    #         val_dir='dataset_224x224/val',
    #         batch_size=32
    # )
    # class_name, confidence = detector.predict_image('test_image_224x224.jpg')
    # print(f"Prediction: {class_name} (Confidence: {confidence:.2%})")
