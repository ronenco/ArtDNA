"""
Script to fetch raw data from Hugging Face dataset.
"""

"""
# NOTE:
# If you want to download private Hugging Face datasets or authenticate the request,
# create a file named `tokenOption.py` in the same folder as this script and define:
#     HF_TOKEN = "your_huggingface_token_here"
#
# If `tokenOption.py` is missing, the script will download anonymously.
"""

from pathlib import Path
from huggingface_hub import snapshot_download

try:
    from tokenOption import HF_TOKEN
except ImportError:
    HF_TOKEN = None

# Download the dataset
output_dir = Path(".")
output_dir.mkdir(parents=False, exist_ok=True)

print("Downloading ArtDNA dataset from Hugging Face...")
snapshot_download(
    repo_id="alonttal/ArtDNA",
    repo_type="dataset",
    local_dir=str(output_dir),
    local_dir_use_symlinks=False,
    token=HF_TOKEN if HF_TOKEN else None
)

print(f"Download complete! Files saved to {output_dir.absolute()}")
