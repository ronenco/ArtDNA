from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.common.ml_utils import set_global_seed

class DatasetPreparation:
    def __init__(self, source_dirs, output_dir, img_size=(224, 224),
                 train_ratio=0.8, random_seed=42):
        """
        Initialize dataset preparation

        Args:
            source_dirs: Dictionary with class names as keys and source directories as values
                        e.g., {'midjourney': 'path/to/midjourney', 'dalle': 'path/to/dalle'}
            output_dir: Directory where organized dataset will be saved
            img_size: Target image size (width, height)
            train_ratio: Proportion of data for training (default: 0.8 = 80%)
            random_seed: Random seed for reproducibility
        """
        self.source_dirs = source_dirs
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        self.random_seed = random_seed
        set_global_seed(random_seed)

        # Validate ratios
        if not (0.0 <= train_ratio <= 1.0):  # Allow small floating point errors
            raise ValueError(f"Train ratio must be between 0.0 and 1.0 (got {train_ratio})")

        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

        np.random.seed(random_seed)

    def get_image_files(self, directory):
        """
        Recursively get all image files from a directory

        Args:
            directory: Source directory path

        Returns:
            List of image file paths
        """
        image_files = []
        directory = Path(directory)

        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist!")
            return image_files

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)

        return image_files

    def resize_and_save_image(self, src_path, dst_path, size=(224, 224)):
        """
        Resize and save an image

        Args:
            src_path: Source image path
            dst_path: Destination image path
            size: Target size (width, height)
        """
        try:
            # Open and convert image to RGB (handles various formats including RGBA)
            img = Image.open(src_path).convert('RGB')

            # Resize with high-quality resampling
            img_resized = img.resize(size, Image.LANCZOS)

            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as JPEG with good quality
            img_resized.save(dst_path, 'JPEG', quality=95)

            return True
        except Exception as e:
            print(f"Error processing {src_path}: {str(e)}")
            return False

    def create_directory_structure(self):
        """Create the train/val directory structure"""
        if 0.0 < self.train_ratio < 1.0:
            splits = ['train', 'val']

            for split in splits:
                for class_name in self.source_dirs.keys():
                    dir_path = self.output_dir / split / class_name
                    dir_path.mkdir(parents=True, exist_ok=True)
        else:
            for class_name in self.source_dirs.keys():
                dir_path = self.output_dir / class_name
                dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Created directory structure in {self.output_dir}")

    def prepare_dataset(self):
        """
        Main function to prepare the dataset:
        1. Collect all images
        2. Split into train/val
        3. Resize and save to appropriate directories
        """
        print("=" * 60)
        print("DATASET PREPARATION")
        print("=" * 60)

        # Create directory structure
        self.create_directory_structure()

        # Process each class
        all_stats = {}

        for class_name, source_dir in self.source_dirs.items():
            print(f"\nProcessing class: {class_name}")
            print("-" * 60)

            # Get all image files
            image_files = self.get_image_files(source_dir)
            print(f"Found {len(image_files)} images in {source_dir}")

            if len(image_files) == 0:
                print(f"Warning: No images found for class {class_name}!")
                continue

            # Split into train and val
            if 0.0 < self.train_ratio < 1.0:
                train_files, val_files = train_test_split(
                    image_files,
                    train_size=self.train_ratio,
                    random_state=self.random_seed,
                    shuffle=True
                )
            elif self.train_ratio == 0.0:
                # All images go to "val" (useful for test-set style exports)
                train_files, val_files = [], image_files
            elif self.train_ratio == 1.0:
                # All images go to "train"
                train_files, val_files = image_files, []
            else:
                raise ValueError(f"Unsupported train_ratio={self.train_ratio}")
            
            print(f"Split: {len(train_files)} train, {len(val_files)} val")

            # Process training images
            print(f"Resizing and saving training images...")
            train_success = 0
            for img_path in tqdm(train_files, desc=f"Train {class_name}"):
                if (self.train_ratio == 1 ):
                    dst_path = self.output_dir / class_name / f"{img_path.stem}.jpg"
                else:
                    dst_path = self.output_dir / 'train' / class_name / f"{img_path.stem}.jpg"
                if self.resize_and_save_image(img_path, dst_path, self.img_size):
                    train_success += 1

            # Process validation images
            print(f"Resizing and saving validation images...")
            val_success = 0
            for img_path in tqdm(val_files, desc=f"Val {class_name}"):
                if (self.train_ratio == 0):
                    dst_path = self.output_dir / class_name / f"{img_path.stem}.jpg"
                else:
                    dst_path = self.output_dir / 'val' / class_name / f"{img_path.stem}.jpg"
                if self.resize_and_save_image(img_path, dst_path, self.img_size):
                    val_success += 1

            # Store statistics
            all_stats[class_name] = {
                'total': len(image_files),
                'train': train_success,
                'val': val_success,
                'failed': len(image_files) - train_success - val_success
            }

        # Print final statistics
        self.print_statistics(all_stats)

        return all_stats

    def print_statistics(self, stats):
        """Print dataset statistics"""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        total_train = 0
        total_val = 0
        total_failed = 0

        for class_name, class_stats in stats.items():
            print(f"\n{class_name.upper()}:")
            print(f"  Total images:      {class_stats['total']}")
            print(f"  Training set:      {class_stats['train']}")
            print(f"  Validation set:    {class_stats['val']}")
            if class_stats['failed'] > 0:
                print(f"  Failed to process: {class_stats['failed']}")

            total_train += class_stats['train']
            total_val += class_stats['val']
            total_failed += class_stats['failed']

        print(f"\nTOTAL:")
        print(f"  Training set:      {total_train}")
        print(f"  Validation set:    {total_val}")
        if total_failed > 0:
            print(f"  Failed to process: {total_failed}")

        print("\n" + "=" * 60)
        print(f"Dataset saved to: {self.output_dir}")
        print("=" * 60)

    def verify_dataset(self):
        """Verify the created dataset"""
        print("\n" + "=" * 60)
        print("DATASET VERIFICATION")
        print("=" * 60)
        if ((self.train_ratio == 1) or (self.train_ratio == 0)):
            for class_dir in self.output_dir.iterdir():
                if class_dir.is_dir():
                    images = list(class_dir.glob('*.jpg'))
                    print(f"  {class_dir.name}: {len(images)} images")

                    # Check a sample image size
                    if images:
                        sample_img = Image.open(images[0])
                        print(f"    Sample image size: {sample_img.size}")
        else:
            for split in ['train', 'val']:
                print(f"\n{split.upper()}:")
                split_dir = self.output_dir / split

                if not split_dir.exists():
                    print(f"  Directory not found!")
                    continue

                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        images = list(class_dir.glob('*.jpg'))
                        print(f"  {class_dir.name}: {len(images)} images")

                        # Check a sample image size
                        if images:
                            sample_img = Image.open(images[0])
                            print(f"    Sample image size: {sample_img.size}")

def print_structure(root_path, title):
    root = Path(root_path)
    print(f"\n{title}: {root.resolve()}")
    if not root.exists():
        print("  (Directory does not exist)")
        return
    for split in ["train", "val"]:
        split_dir = root / split
        print(f"  {split}/ -> {split_dir.resolve()}")
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    print(f"    {class_dir.name}/ ({len(list(class_dir.glob('*.jpg')))} images)")
        else:
            print("    (Missing)")



if __name__ == "__main__":
    # Example 1: Basic usage with two classes
    print("EXAMPLE 1: Two classes (Midjourney and DALL-E)")
    print("-" * 60)

    source_directories = {
        'midjourney': 'raw_data/midjourney',
        'dalle': 'raw_data/dalle'
    }
    test_directories = {
        'midjourney' : 'raw_test/midjourney',
        'dalle': 'raw_test/dalle'
    }

    # Prepration for efficientNet:
    prep = DatasetPreparation(
        source_dirs=source_directories,
        output_dir='./dataset/dataset_224x224',
        img_size=(224, 224),
        train_ratio=0.8,
    )

    # Prepration for Xception:
    prep2 = DatasetPreparation(
        source_dirs=source_directories,
        output_dir='./dataset/dataset_299x299',
        img_size=[299, 299],
        train_ratio=0.8,
    )

    # Test Set generation:

    # For regular based:
    testPrep = DatasetPreparation(
        source_dirs= test_directories,
        output_dir='./dataset/testset_224x224',
        img_size=(224, 224),
        train_ratio=0,
    )

    # For xception:
    testPrep2 = DatasetPreparation(
        source_dirs= test_directories,
        output_dir='./dataset/testset_299x299',
        img_size=(299, 299),
        train_ratio=0,
    )


    # Uncomment to run:
    stats = prep.prepare_dataset()
    prep.verify_dataset()

    stats2 = prep2.prepare_dataset()
    prep2.verify_dataset()

    print("\nEXAMPLE 2: Test set generation (224x224 and 299x299)")
    print("-" * 60)

    testStats = testPrep.prepare_dataset()
    testPrep.verify_dataset()

    testStats2 = testPrep2.prepare_dataset()
    testPrep2.verify_dataset()

    print("\n" + "=" * 60)
    print("DATASET & TESTSET LOCATIONS")
    print("=" * 60)

    print_structure("./dataset/dataset_224x224", "Training Dataset 224x224")
    print_structure("./dataset/dataset_299x299", "Training Dataset 299x299")
    print_structure("./dataset/testset_224x224", "Testset 224x224")
    print_structure("./dataset/testset_299x299", "Testset 299x299")

    print("\nSummary complete.")
    print("=" * 60)
