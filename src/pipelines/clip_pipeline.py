import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip
from tqdm import tqdm

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.ml_utils import set_global_seed, plot_training_history_torch


# ---------- Config ----------

MODEL_NAME = "ViT-B-32"         # CLIP backbone
PRETRAIN_DATASET = "laion2b_s34b_b79k"  # pretraining configuration
BATCH_SIZE = 32
TEST_BATCH_SIZE = 20
EPOCHS = 10
SEED = 42
TRAINABLE_CLIP_LAYERS = 3 # number of CLIP visual layers to unfreeze (0 = freeze all)
LR = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = Path("dataset/dataset_224x224")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TESTSET_ROOT = Path("dataset/testset_224x224")
MODEL_SAVE_PATH = "clip_midjourney_vs_dalle_best.pt"


# ---------- Dataset ----------

def build_dataloaders(preprocess_transform, train_dir: Path = TRAIN_DIR, val_dir: Path = VAL_DIR, batch_size: int = BATCH_SIZE):
    """
    Use torchvision.datasets.ImageFolder so that:
      class 0 = dalle
      class 1 = midjourney
    (based on subfolder names in TRAIN_DIR / VAL_DIR)
    """

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=preprocess_transform
    )

    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=preprocess_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE == "cuda")
    )

    print("Classes:", train_dataset.classes)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    return train_loader, val_loader

def build_test_loader(preprocess_transform, test_dir: Path = TESTSET_ROOT, batch_size: int = BATCH_SIZE):
    """
    Build a DataLoader for the held-out test set.

    Expects the same folder structure as train/val:
        test_dir / class_name / *.png
    """
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=preprocess_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
    )

    print("Test classes:", test_dataset.classes)
    print("Test samples:", len(test_dataset))

    return test_loader


# ---------- Model: CLIP image encoder + small classifier ----------

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, trainable_layers=0):
        super().__init__()
        self.clip_model = clip_model
        self.finetune_backbone = trainable_layers > 0
        # Freeze all CLIP parameters by default
        for p in self.clip_model.parameters():
            p.requires_grad = False

        if trainable_layers > 0:
            # Unfreeze the last N transformer blocks (and ln_post) of the visual backbone
            if hasattr(self.clip_model, "visual") and hasattr(self.clip_model.visual, "transformer"):
                blocks = self.clip_model.visual.transformer.resblocks
                total_blocks = len(blocks)
                n = min(trainable_layers, total_blocks)
                for block in blocks[total_blocks - n :]:
                    for p in block.parameters():
                        p.requires_grad = True

                # Also unfreeze final layer norm / ln_post if present
                if hasattr(self.clip_model.visual, "ln_post"):
                    for p in self.clip_model.visual.ln_post.parameters():
                        p.requires_grad = True

        # Simple linear head on top of CLIP embedding
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # binary logits
        )

    def forward(self, images):
        # CLIP image encoder expects already-preprocessed tensors
        if self.finetune_backbone:
            features = self.clip_model.encode_image(images)
        else:
            with torch.no_grad():
                features = self.clip_model.encode_image(images)

        # Optionally normalize features
        features = features / features.norm(dim=-1, keepdim=True)

        logits = self.classifier(features)
        return logits.squeeze(1)


def generate_report(model, loader, device, output_path: str = "clip_confusion_matrix.png", desc: str = "Report"):
    """
    Generate a classification report, confusion matrix, and summary metrics
    (accuracy and F1) for a given model and DataLoader.

    Returns a dict with the main metrics for optional downstream use.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()

            all_labels.append(labels.numpy())
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    class_names = loader.dataset.classes

    # Summary metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print("\nSummary metrics:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  F1 (macro):     {f1_macro:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "support": len(all_labels),
    }


# ---------- Training / Evaluation loops ----------

def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing: float = 0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)  # BCEWithLogitsLoss expects float labels (0/1)

        if label_smoothing > 0.0:
            smooth = label_smoothing
            # For binary labels, smooth towards 0.5
            labels = labels * (1.0 - smooth) + 0.5 * smooth

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds.cpu() == labels.cpu().long()).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == labels.cpu().long()).sum().item()
            total += images.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def run_pipeline(
    model_name: str = MODEL_NAME,
    pretrained_dataset: str = PRETRAIN_DATASET,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    trainable_clip_layers: int = TRAINABLE_CLIP_LAYERS,
    dataset_root: Path = DATASET_ROOT,
    model_save_path: str = MODEL_SAVE_PATH,
    seed: int = SEED,
    device: str = DEVICE,
    label_smoothing: float = 0.0,
):
    """Run the full CLIP-based pipeline end-to-end.

    All arguments have sensible defaults taken from the config section,
    but can be overridden when calling this function.
    """
    # Set seeds for reproducibility
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Using device: {device}")

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    # Load CLIP model & preprocess
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_dataset,
        device=device
    )

    embed_dim = clip_model.visual.output_dim
    print(f"CLIP visual embed_dim: {embed_dim}")

    # Build dataloaders with CLIP's preprocessing
    train_loader, val_loader = build_dataloaders(
        clip_preprocess,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
    )

    # Compute class balance and pos_weight for BCEWithLogitsLoss
    targets = np.array(train_loader.dataset.targets)
    num_pos = (targets == 1).sum()
    num_neg = (targets == 0).sum()
    if num_pos > 0 and num_neg > 0:
        pos_weight_value = num_neg / num_pos
        if (device == "mps"):
            pos_weight_value = pos_weight_value.astype(np.float32)
        pos_weight = torch.tensor([pos_weight_value], device=device)
        print(f"Using pos_weight={pos_weight_value:.4f} for BCEWithLogitsLoss (neg={num_neg}, pos={num_pos})")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("Warning: could not compute class weights (one of the classes has zero samples). Using unweighted loss.")
        criterion = nn.BCEWithLogitsLoss()

    # Wrap CLIP image encoder in classifier
    model = CLIPClassifier(
        clip_model,
        embed_dim,
        trainable_layers=trainable_clip_layers
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_smoothing=label_smoothing,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # Append to data
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_trained_model(model, model_save_path)
            print(f"✅ New best model saved (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    
    # Plot the training history:
    plot_training_history_torch(
        train_losses, val_losses, train_accs, val_accs,
        title_prefix="CLIP"
    )

    # Reload the best model (by validation accuracy) from disk so downstream
    # evaluation and report use the best checkpoint.
    best_model, test_loader, test_loss, test_acc = evaluate_on_testset()
    
    return best_model, val_loader


def save_trained_model(model: nn.Module, path: str = MODEL_SAVE_PATH):
    """Save the full CLIPClassifier (backbone + head) state_dict to a file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(model.state_dict(), path)
    print(f"Model state_dict saved to: {path}")


def load_trained_model(
    model_name: str = MODEL_NAME,
    pretrained_dataset: str = PRETRAIN_DATASET,
    trainable_clip_layers: int = TRAINABLE_CLIP_LAYERS,
    model_path: str = MODEL_SAVE_PATH,
    device: str = DEVICE,
) -> nn.Module:
    """Recreate the CLIPClassifier architecture and load weights from `model_path`.

    This assumes the same CLIP backbone configuration (model_name, pretrained_dataset)
    that was used for training.
    """
    # Recreate CLIP backbone
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_dataset,
        device=device,
    )

    embed_dim = clip_model.visual.output_dim

    # Wrap it in our classifier and load weights
    model = CLIPClassifier(
        clip_model,
        embed_dim,
        trainable_layers=trainable_clip_layers,
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded trained model from: {model_path}")
    return model

def evaluate_on_testset(
    model_path: str = MODEL_SAVE_PATH,
    testset_root: Path = TESTSET_ROOT,
    model_name: str = MODEL_NAME,
    pretrained_dataset: str = PRETRAIN_DATASET,
    trainable_clip_layers: int = TRAINABLE_CLIP_LAYERS,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
):
    """Load a trained model and evaluate it on a held-out test set.

    Expects testset_root to have the same folder structure as train/val:
        testset_root / class_name / *.png
    """
    if not testset_root.exists():
        raise FileNotFoundError(f"Testset directory not found: {testset_root}")

    # Get CLIP preprocess for the given backbone
    _, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_dataset,
        device=device,
    )

    # Build test loader
    test_loader = build_test_loader(
        clip_preprocess,
        test_dir=testset_root,
        batch_size=batch_size,
    )

    # Recreate model and load weights
    model = load_trained_model(
        model_name=model_name,
        pretrained_dataset=pretrained_dataset,
        trainable_clip_layers=trainable_clip_layers,
        model_path=model_path,
        device=device,
    )

    # Use unweighted loss for reporting on the test set
    criterion = nn.BCEWithLogitsLoss()
    print(f"\nLoading model using test loader: {testset_root}")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # Detailed classification report, metrics & confusion matrix (saved as image)
    test_metrics = generate_report(
        model,
        test_loader,
        device,
        output_path="clip_confusion_matrix_test.png",
        desc="Test report",
    )

    print("\n[TEST SUMMARY]")
    print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):     {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {test_metrics['f1_weighted']:.4f}")
    
    return model, test_loader, test_loss, test_acc


# ---------- Main ----------

def main():
    model, val_loader = run_pipeline()
    return model, val_loader


if __name__ == "__main__":
    # Run the pipeline to train and load the BEST validation model
    model, val_loader = main()

    # Validation report (best model), including confusion matrix image
    print("\n[VALIDATION SUMMARY - BEST MODEL]")
    val_metrics = generate_report(
        model,
        val_loader,
        DEVICE,
        output_path="clip_confusion_matrix_val.png",
        desc="Val report",
    )

    print("\n[VALIDATION METRICS]")
    print(f"  Accuracy:       {val_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):     {val_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {val_metrics['f1_weighted']:.4f}")

    # Run on the held-out test set using the same best checkpoint
