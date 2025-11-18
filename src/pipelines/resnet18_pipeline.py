# ============================================
# ArtDNA – Image Classification (ResNet18)
# PyTorch training pipeline
# ============================================

from pathlib import Path
from typing import Tuple, List

import random
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

from src.common.ml_utils import set_global_seed, plot_training_history_torch


# ---------- Config (defaults – ניתן לדרוס דרך main.py) ----------

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
LABEL_SMOOTHING = 0.1
TRAINABLE_BASE_LAYERS = 5
SEED = 42

DATASET_ROOT = Path("dataset/dataset_224x224")
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


# ---------- Utils ----------

def set_seed_full(seed: int = 42):
    """
    Set random seeds for full reproducibility.
    משתמש גם ב־set_global_seed של הפרויקט וגם בהגדרות של PyTorch.
    """
    set_global_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For full determinism (can slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(
    dataset_root: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    device: str = DEVICE,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch DataLoaders for train and val folders.

    Expected folder structure:
        dataset_root/
            train/
                class_1/
                class_2/
                ...
            val/
                class_1/
                class_2/
                ...
    """

    # ImageNet-style normalization (for pretrained ResNet)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)

    pin_memory = (device == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("[ResNet18 Pipeline] Classes:", train_dataset.classes)
    print("[ResNet18 Pipeline] Train samples:", len(train_dataset))
    print("[ResNet18 Pipeline] Val   samples:", len(val_dataset))

    return train_loader, val_loader, train_dataset.classes


def build_resnet18(
    num_classes: int,
    trainable_base_layers: int = TRAINABLE_BASE_LAYERS,
) -> nn.Module:
    """
    Build a ResNet18 model with a custom classification head.
    Only the last 'trainable_base_layers' modules of the backbone will be unfrozen.
    """

    # Load pretrained ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 'trainable_base_layers' children modules
    # e.g. if trainable_base_layers=5 we unfreeze the last 5 blocks (including fc)
    children = list(model.children())
    if trainable_base_layers > 0:
        for child in children[-trainable_base_layers:]:
            for param in child.parameters():
                param.requires_grad = True

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


# ---------- Main pipeline API main.py ----------

def run_pipeline(
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    label_smoothing: float = LABEL_SMOOTHING,
    trainable_base_layers: int = TRAINABLE_BASE_LAYERS,
    dataset_root: Path = DATASET_ROOT,
    seed: int = SEED,
):
    """
    Main ResNet18 training pipeline.

    Designed to be called from main.py, e.g.:

        model, (train_loader, val_loader) = run_pipeline(
            batch_size=..., epochs=..., lr=..., label_smoothing=...,
            trainable_base_layers=..., dataset_root=..., seed=...
        )
    """

    # ---- Reproducibility ----
    set_seed_full(seed)

    # ---- Device ----
    device_str = DEVICE  # from global config
    device = torch.device(device_str)
    print(f"[ResNet18 Pipeline] Using device: {device_str}")

    # ---- Data ----
    print(f"[ResNet18 Pipeline] Loading data from: {dataset_root}")
    train_loader, val_loader, class_names = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=4,
        device=device_str,
    )
    num_classes = len(class_names)
    print(f"[ResNet18 Pipeline] Classes: {class_names} (num_classes={num_classes})")

    # ---- Model ----
    model = build_resnet18(
        num_classes=num_classes,
        trainable_base_layers=trainable_base_layers,
    )
    model = model.to(device)

    # ---- Loss & Optimizer ----
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_acc = 0.0
    best_model_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # ---- Training Loop ----
    for epoch in range(1, epochs + 1):
        print(f"\n[ResNet18 Pipeline] Epoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[ResNet18 Pipeline] Train  Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"[ResNet18 Pipeline] Val    Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Save best model 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"[ResNet18 Pipeline] ✅ New best model (val_acc={best_val_acc:.4f})")

    # Load best weights back into the model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n[ResNet18 Pipeline] Training finished. Best val_acc = {best_val_acc:.4f}")

    
    plot_training_history_torch(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        title_prefix="ResNet18",
    )

    # Return model and loaders in a similar format to other pipelines
    return model, (train_loader, val_loader)



if __name__ == "__main__":
    run_pipeline()
