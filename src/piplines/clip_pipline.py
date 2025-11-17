import os
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip
from tqdm import tqdm

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Config ----------

MODEL_NAME = "ViT-B-32"         # CLIP backbone
PRETRAIN_DATASET = "laion2b_s34b_b79k"  # pretraining configuration
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
TRAINABLE_CLIP_LAYERS = 3 # number of CLIP visual layers to unfreeze (0 = freeze all)
LR = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = Path("dataset/dataset_224x224")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
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


def generate_report(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Report", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()

            all_labels.append(labels.numpy())
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    class_names = loader.dataset.classes

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("clip_confusion_matrix.png")
    plt.close()


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
    random.seed(seed)
    np.random.seed(seed)
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

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    return model, val_loader


# ---------- Main ----------

def main():
    model, val_loader = run_pipeline()
    return model, val_loader


if __name__ == "__main__":
    # Run the Pipline to generate model
    model, val_loader = main()

    #Generate report:
    generate_report(model, val_loader, DEVICE)
