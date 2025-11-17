from pathlib import Path

from src.piplines.xception_pipline import run_pipeline as run_xception_pipeline
from src.piplines.efficeintnet_pipline import run_pipeline as run_efficientnet_pipeline
from src.piplines.clip_pipline import run_pipeline as run_clip_pipeline

import argparse

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

    parser.add_argument("--model-type", type=str, default="xception",
                        choices=["xception", "efficientnet", "clip"],
                        help="Which model pipeline to run")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--trainable-base-layers", type=int, default=5,
                        help="Used for Xception/EfficientNet fine-tuning")
    parser.add_argument("--trainable-clip-layers", type=int, default=2,
                        help="Used for CLIP fine-tuning")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Optional override for dataset root folder. "
                             "If not provided, a default is chosen per model type.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Choose dataset root: either override from CLI, or sensible default per model type
    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root)
    else:
        if args.model_type in ("efficientnet", "clip"):
            dataset_root = DATASET_224
        elif args.model_type == "xception":
            dataset_root = DATASET_299
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {args.model_type}")

    # Ensure dataset exists and contains images before running the pipeline
    ensure_dataset_has_images(dataset_root)

    if args.model_type == "xception":
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

    elif args.model_type == "efficientnet":
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

    elif args.model_type == "clip":
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

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {args.model_type}")


if __name__ == "__main__":
    main()