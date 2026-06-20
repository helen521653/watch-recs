import subprocess
import urllib.request
from pathlib import Path

_BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
    "/benchmark/5core/last_out_w_his"
)

_SPLIT_FILES = [
    "Movies_and_TV.train.csv.gz",
    "Movies_and_TV.valid.csv.gz",
    "Movies_and_TV.test.csv.gz",
]


def download_data(data_dir: Path = Path(".")) -> None:
    """Download Amazon Reviews 2023 Movies_and_TV 5-core pre-split files.

    Source: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/
    """
    if all((data_dir / f).exists() for f in _SPLIT_FILES):
        print("Data files already exist, skipping download.")
        return

    for filename in _SPLIT_FILES:
        dest = data_dir / filename
        if dest.exists():
            print(f"{filename} already exists, skipping.")
            continue
        url = f"{_BASE_URL}/{filename}"
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)

    print("Download complete.")


def pull_dvc(remote: str = "data-remote") -> None:
    """Pull data files from DVC remote storage."""
    subprocess.run(["dvc", "pull", "--remote", remote], check=True)


def pull_models(remote: str = "models-remote") -> None:
    """Pull model checkpoints from DVC remote storage."""
    subprocess.run(["dvc", "pull", "--remote", remote], check=True)
