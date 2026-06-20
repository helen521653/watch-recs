import subprocess
from pathlib import Path

import gdown

_GDRIVE_IDS = {
    "Movies_and_TV.train.csv.gz": "10rNbtiXlK8HHgto2PTmQu45vmO8QBcBR",
    "Movies_and_TV.valid.csv.gz": "1Z89RFVj8hcZlx52c6SOjfYFLZR7_gD5C",
    "Movies_and_TV.test.csv.gz": "1UlHijM2ubc5NhRFlkaJhCqRzGLPZXvZa",
}

_SPLIT_FILES = list(_GDRIVE_IDS.keys())


def download_data(data_dir: Path = Path(".")) -> None:
    """Download Amazon Reviews 2023 Movies_and_TV 5-core pre-split files from Google Drive."""
    if all((data_dir / f).exists() for f in _SPLIT_FILES):
        print("Data files already exist, skipping download.")
        return

    for filename, file_id in _GDRIVE_IDS.items():
        dest = data_dir / filename
        if dest.exists():
            print(f"{filename} already exists, skipping.")
            continue
        print(f"Downloading {filename}...")
        gdown.download(id=file_id, output=str(dest), quiet=False)

    print("Download complete.")


def pull_dvc(remote: str = "data-remote") -> None:
    """Pull data files from DVC remote storage."""
    subprocess.run(["dvc", "pull", "--remote", remote], check=True)


def pull_models(remote: str = "models-remote") -> None:
    """Pull model checkpoints from DVC remote storage."""
    subprocess.run(["dvc", "pull", "--remote", remote], check=True)
