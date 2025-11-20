"""
Visualization utilities for embedding spaces.

Provides helper functions to generate PCA and t-SNE scatter plots for
class-level inspection of latent features produced by the different models,
plus a CLI for quick inspection of datasets structured as:

dataset/
└── dataset_224x224/
    ├── train/
    └── val/
        ├── dalle/
        └── midjourney/
"""

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
LabelVector = Union[Sequence[int], np.ndarray]
LabelNames = Optional[Mapping[int, str]]
FigureSize = Tuple[float, float]


def _prepare_labels(labels: LabelVector, label_names: LabelNames = None):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"'labels' must be 1D, got shape {labels.shape}.")

    if label_names is None:
        label_text = labels.astype(str)
    else:
        label_text = np.array([label_names.get(int(idx), str(idx)) for idx in labels])
    return labels, label_text


def _scatter_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    label_text: np.ndarray,
    title: str,
    figsize: FigureSize,
    save_path: Optional[Union[str, Path]] = None,
):
    if embedding.shape[1] != 2:
        raise ValueError("Embedding must be 2D (n_samples, 2).")

    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(np.unique(labels)))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=label_text,
        palette=palette,
        alpha=0.8,
        s=45,
        edgecolor="none",
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class", loc="best", fontsize="small")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


def plot_pca_embeddings(
    features: ArrayLike,
    labels: LabelVector,
    *,
    n_components: int = 2,
    label_names: LabelNames = None,
    figsize: FigureSize = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    title: str = "PCA Embedding",
    standardize: bool = True,
):
    """
    Project high-dimensional features into a lower-dimensional space using PCA
    and plot the resulting 2D scatter plot.
    """

    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"'features' must be 2D, got shape {features.shape}.")

    if standardize:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

    labels_numeric, label_text = _prepare_labels(labels, label_names)

    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(features)

    if n_components >= 2:
        embedding_2d = embedding[:, :2]
    else:
        raise ValueError("PCA embeddings need at least 2 components for plotting.")

    explained_var = pca.explained_variance_ratio_[:2] * 100
    title_full = f"{title} (Explained: {explained_var[0]:.1f}% / {explained_var[1]:.1f}%)"

    _scatter_plot(embedding_2d, labels_numeric, label_text, title_full, figsize, save_path)


def plot_tsne_embeddings(
    features: ArrayLike,
    labels: LabelVector,
    *,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    n_iter: int = 1000,
    label_names: LabelNames = None,
    figsize: FigureSize = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    title: str = "t-SNE Embedding",
    random_state: int = 42,
):
    """
    Generate a 2D t-SNE projection for visualizing class separability.
    """

    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"'features' must be 2D, got shape {features.shape}.")

    labels_numeric, label_text = _prepare_labels(labels, label_names)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init="pca",
        random_state=random_state,
    )
    embedding = tsne.fit_transform(features)

    _scatter_plot(embedding, labels_numeric, label_text, title, figsize, save_path)


def _load_dataset(
    dataset_root: Union[str, Path],
    subset: str = "train",
    *,
    class_names: Optional[Sequence[str]] = None,
    sample_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, LabelNames]:
    """
    Load images from the dataset directory, flattening them into feature vectors.
    """

    dataset_root = Path(dataset_root)
    subset_dir = dataset_root / subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if class_names is None:
        class_names = sorted([p.name for p in subset_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class subdirectories found in {subset_dir}")

    features = []
    labels = []
    label_map = {}

    for label_idx, class_name in enumerate(class_names):
        class_dir = subset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        label_map[label_idx] = class_name
        image_paths = [
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts
        ]
        if not image_paths:
            raise RuntimeError(f"No images found in {class_dir}")

        if sample_limit is not None:
            image_paths = image_paths[:sample_limit]

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32).reshape(-1)
            features.append(arr)
            labels.append(label_idx)

    if not features:
        raise RuntimeError("Dataset loader did not collect any samples.")

    return np.vstack(features), np.array(labels, dtype=np.int64), label_map


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embeddings via PCA or t-SNE.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_224x224",
        help="Path to dataset root containing a 'train' directory.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="Dataset subset to visualize (default: train).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "tsne"],
        default="tsne",
        help="Dimensionality reduction technique to use.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional file path to save the plot image.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=500,
        help="Max number of samples per class to load (default: all).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (only used for method=tsne).",
    )
    parser.add_argument(
        "--learning-rate",
        type=str,
        default="auto",
        help="t-SNE learning rate (only used for method=tsne).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of PCA components to compute (min 2).",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization before PCA.",
    )
    return parser.parse_args()


def main():
    # Example usage:
    #   python src/common/visualization.py
    #       Runs default t-SNE on dataset/dataset_224x224/train (dalle vs midjourney)
    #   python src/common/visualization.py --method pca --save-path plots/pca.png
    #       Projects with PCA and writes the figure to plots/pca.png

    args = parse_args()
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        # allow shorthand like "dataset_224x224"
        potential = Path("dataset") / args.dataset_dir
        if potential.exists():
            dataset_path = potential
        else:
            raise FileNotFoundError(
                f"Dataset directory not found: {args.dataset_dir}. "
                "Provide an absolute path or ensure it exists relative to the project root."
            )

    print(f"[viz] Loading subset='{args.subset}' from '{dataset_path}' using method='{args.method.upper()}'")

    features, labels, label_map = _load_dataset(
        dataset_path,
        subset=args.subset,
        class_names=["dalle", "midjourney"],
        sample_limit=args.sample_limit,
    )

    print(
        f"[viz] Loaded {features.shape[0]} samples "
        f"({len(label_map)} classes: {', '.join(label_map.values())}) "
        f"with feature dim {features.shape[1]}"
    )

    title = f"{args.method.upper()} - {args.subset} ({args.dataset_dir})"

    if args.method == "pca":
        plot_pca_embeddings(
            features,
            labels,
            n_components=args.n_components,
            label_names=label_map,
            save_path=args.save_path,
            title=title,
            standardize=not args.no_standardize,
        )
    else:
        plot_tsne_embeddings(
            features,
            labels,
            label_names=label_map,
            save_path=args.save_path,
            title=title,
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()

