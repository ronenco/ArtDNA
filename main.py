from pathlib import Path

from src.piplines.xception_pipline import run_pipeline as run_xception_pipeline
from src.piplines.efficeintnet_pipline import run_pipeline as run_efficientnet_pipeline
from src.piplines.clip_pipline import run_pipeline as run_clip_pipeline

import argparse

# Hyperparameters will now come from CLI arguments

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

    return parser.parse_args()

def main():
    args = parse_args()

    if args.model_type == "xception":
        print("Running Xception pipeline...")
        model, (train_gen, val_gen) = run_xception_pipeline(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            label_smoothing=args.label_smoothing,
            trainable_base_layers=args.trainable_base_layers,
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
            seed=args.seed,
        )
        return model, val_loader

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {args.model_type}")


if __name__ == "__main__":
    main()