import argparse
from pathlib import Path
from typing import Optional

from src.common.logging_utils import tee_output
from src.pipelines.clip_pipeline import run_pipeline as run_clip_pipeline
from src.pipelines.efficeintnet_pipeline import run_pipeline as run_efficientnet_pipeline
from src.pipelines.resnet18_pipeline import run_pipeline as run_resnet18_pipeline
from src.pipelines.resnet50_pipeline import run_pipeline as run_resnet50_pipeline
from src.pipelines.xception_pipeline import run_pipeline as run_xception_pipeline
from src.pipelines.mobilenet_pipeline import run_pipeline as run_mobilenet_pipeline

# Hyperparameters will now come from CLI arguments

# Default dataset roots per model type
DATASET_224 = Path("dataset/dataset_224x224")
DATASET_299 = Path("dataset/dataset_299x299")


def ensure_dataset_has_images(dataset_root: Path):
    """Verify that train/val folders exist and contain files. Raise a clear error if not."""
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            f"Expected 'train' and 'val' subdirectories under: {dataset_root}. "
            f"Looked for: '{train_dir}' and '{val_dir}'."
        )

    # Check that there is at least one file in each directory (recursively)
    has_train_files = any(train_dir.rglob("*"))
    has_val_files = any(val_dir.rglob("*"))

    if not has_train_files or not has_val_files:
        missing = []
        if not has_train_files:
            missing.append(str(train_dir))
        if not has_val_files:
            missing.append(str(val_dir))
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "No images found in the expected dataset folders. "
            f"Looked for image files under: {missing_str}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run ArtDNA training pipeline")

    parser.add_argument(
        "--model-type",
        type=str,
        default="xception",
        choices=["xception", "efficientnet", "clip", "resnet18", "resnet50", "mobilenet"],
        help="Which model pipeline to run",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument(
        "--trainable-base-layers",
        type=int,
        default=5,
        help="Used for Xception/EfficientNet/ResNet/MobileNet fine-tuning",
    )
    parser.add_argument(
        "--trainable-clip-layers",
        type=int,
        default=2,
        help="Used for CLIP fine-tuning",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Optional override for dataset root folder. "
            "If not provided, a default is chosen per model type."
        ),
    )

    return parser.parse_args()


def _resolve_dataset_root(model_type: str, override: Optional[str]) -> Path:
    if override is not None:
        return Path(override)

    if model_type in ("efficientnet", "clip", "resnet18", "resnet50", "mobilenet"):
        return DATASET_224
    if model_type == "xception":
        return DATASET_299

    raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")


def main():
    args = parse_args()
    model_type = args.model_type.lower()

    with tee_output(model_type):
        dataset_root = _resolve_dataset_root(model_type, args.dataset_root)

        # Ensure dataset exists and contains images before running the pipeline
        ensure_dataset_has_images(dataset_root)

        if model_type == "xception":
            print("Running Xception pipeline...")
            model, (train_gen, val_gen) = run_xception_pipeline(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                trainable_base_layers=args.trainable_base_layers,
                dataset_root=dataset_root,
                seed=args.seed,
            )
            return model, (train_gen, val_gen)

        if model_type == "efficientnet":
            print("Running EfficientNetB0 pipeline...")
            model, (train_gen, val_gen) = run_efficientnet_pipeline(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                trainable_base_layers=args.trainable_base_layers,
                dataset_root=dataset_root,
                seed=args.seed,
            )
            return model, (train_gen, val_gen)

        if model_type == "clip":
            print("Running CLIP pipeline...")
            model, val_loader = run_clip_pipeline(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                trainable_clip_layers=args.trainable_clip_layers,
                dataset_root=dataset_root,
                seed=args.seed,
            )
            return model, val_loader

        if model_type == "resnet18":
            print("Running ResNet18 pipeline...")
            model, (train_loader, val_loader) = run_resnet18_pipeline(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                trainable_base_layers=args.trainable_base_layers,
                dataset_root=dataset_root,
                seed=args.seed,
            )
            return model, (train_loader, val_loader)

        if model_type == "resnet50":
            print("Running ResNet50 pipeline...")
            model, (train_gen, val_gen) = run_resnet50_pipeline(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                trainable_base_layers=args.trainable_base_layers,
                dataset_root=dataset_root,
                seed=args.seed,
            )
            return model, (train_gen, val_gen)

        if model_type == "mobilenet":
            print("Running MobileNet pipeline...")
            model, (train_gen, val_gen) = run_mobilenet_pipeline(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                trainable_base_layers=args.trainable_base_layers,
                dataset_root=dataset_root,
                seed=args.seed,
            )
            return model, (train_gen, val_gen)

        raise ValueError(f"Unsupported MODEL_TYPE: {args.model_type}")


if __name__ == "__main__":
    main()
