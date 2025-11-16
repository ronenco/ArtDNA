"""
Script to fetch raw data from Hugging Face dataset.
"""

from pathlib import Path
from huggingface_hub import snapshot_download

# Download the dataset
output_dir = Path("./")
output_dir.mkdir(parents=False, exist_ok=True)

print("Downloading ArtDNA dataset from Hugging Face...")
snapshot_download(
    repo_id="alonttal/ArtDNA",
    repo_type="dataset",
    local_dir=str(output_dir),
    local_dir_use_symlinks=False
)

print(f"Download complete! Files saved to {output_dir.absolute()}")

