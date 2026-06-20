from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split Amazon Reviews files (pandas reads .gz natively)."""
    train_df = pd.read_csv(data_dir / "Movies_and_TV.train.csv.gz")
    val_df = pd.read_csv(data_dir / "Movies_and_TV.valid.csv.gz")
    test_df = pd.read_csv(data_dir / "Movies_and_TV.test.csv.gz")
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


class DataPreprocessor:
    """Encodes raw user/item string IDs to dense integer indices.

    Fit on train split only, then transform all splits — so val/test
    use the same mapping.
    """

    def __init__(self, min_positive_rating: int = 4) -> None:
        self.min_positive_rating = min_positive_rating
        self.user2idx: dict[str, int] = {}
        self.item2idx: dict[str, int] = {}
        self._num_users: int = 0
        self._num_items: int = 0

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    def fit(self, train_df: pd.DataFrame) -> "DataPreprocessor":
        """Build user/item vocabularies from train data."""
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        self.user2idx = {uid: idx for idx, uid in enumerate(users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(items)}
        self._num_users = len(users)
        self._num_items = len(items)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoders and binarize ratings. Drops rows with unknown IDs."""
        result = df.copy()
        result["user_idx"] = result["user_id"].map(self.user2idx)
        result["item_idx"] = result["parent_asin"].map(self.item2idx)
        result["label"] = (result["rating"] >= self.min_positive_rating).astype(int)
        result["history_idxs"] = result["history"].apply(self._parse_history)
        return result.dropna(subset=["user_idx", "item_idx"])

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def _parse_history(self, history_str: str) -> set[int]:
        """Parse 'item1 item2 ...' string from history column into item index set.

        Custom code: no library provides this for Amazon Reviews format.
        """
        if pd.isna(history_str) or history_str == "":
            return set()
        raw_ids = str(history_str).strip().split()
        return {self.item2idx[iid] for iid in raw_ids if iid in self.item2idx}


# --------------------------------------------------------------------------- #
# Dataset (training)
# --------------------------------------------------------------------------- #


class InteractionDataset(Dataset):
    """Training dataset with negative sampling.

    Negative sampling is custom code: no standard library provides
    on-the-fly negative sampling for implicit-feedback CF.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_items: int,
        neg_sample_ratio: int = 4,
        seed: int = 42,
    ) -> None:
        self._num_items = num_items
        self._neg_sample_ratio = neg_sample_ratio
        self._rng = np.random.default_rng(seed)

        # full history per user to exclude already-seen items from negatives
        self._user_history: dict[int, set[int]] = (
            df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        positive_df = df[df["label"] == 1]
        self._samples = self._build_samples(positive_df)

    def _build_samples(self, positive_df: pd.DataFrame) -> list[tuple[int, int, float]]:
        user_ids = positive_df["user_idx"].tolist()
        item_ids = positive_df["item_idx"].tolist()

        samples: list[tuple[int, int, float]] = []
        for user_idx, item_idx in tqdm(
            zip(user_ids, item_ids), total=len(user_ids), desc="Negative sampling"
        ):
            samples.append((user_idx, item_idx, 1.0))

            history = self._user_history.get(user_idx, set())
            neg_count = 0
            while neg_count < self._neg_sample_ratio:
                neg_item = int(self._rng.integers(0, self._num_items))
                if neg_item not in history:
                    samples.append((user_idx, neg_item, 0.0))
                    neg_count += 1

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, item_idx, label = self._samples[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


# --------------------------------------------------------------------------- #
# LightningDataModule — Lightning-way to wrap all data logic
# --------------------------------------------------------------------------- #


class RecsDataModule(L.LightningDataModule):
    """Wraps loading, preprocessing, and DataLoaders in one Lightning object."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1024,
        neg_sample_ratio: int = 4,
        eval_neg_sample_ratio: int = 49,
        num_workers: int = 4,
        seed: int = 42,
        min_positive_rating: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.preprocessor = DataPreprocessor(min_positive_rating=min_positive_rating)

    @property
    def num_users(self) -> int:
        return self.preprocessor.num_users

    @property
    def num_items(self) -> int:
        return self.preprocessor.num_items

    @property
    def train_df(self) -> pd.DataFrame:
        return self._train_df

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "_train_dataset"):
            return

        print("Loading data files...")
        train_raw, val_raw, test_raw = load_splits(self.data_dir)
        print(
            f"Train: {len(train_raw)} rows, Val: {len(val_raw)}, Test: {len(test_raw)}"
        )

        print("Encoding...")
        train_enc = self.preprocessor.fit_transform(train_raw)
        val_enc = self.preprocessor.transform(val_raw)
        test_enc = self.preprocessor.transform(test_raw)
        self._train_df = train_enc
        print(
            f"Users: {self.preprocessor.num_users}, Items: {self.preprocessor.num_items}"
        )

        print("Building train dataset...")
        self._train_dataset = InteractionDataset(
            train_enc,
            num_items=self.preprocessor.num_items,
            neg_sample_ratio=self.hparams.neg_sample_ratio,
            seed=self.hparams.seed,
        )
        print("Building val dataset...")
        self._val_dataset = InteractionDataset(
            val_enc,
            num_items=self.preprocessor.num_items,
            neg_sample_ratio=self.hparams.eval_neg_sample_ratio,
            seed=self.hparams.seed,
        )
        print("Building test dataset...")
        self._test_dataset = InteractionDataset(
            test_enc,
            num_items=self.preprocessor.num_items,
            neg_sample_ratio=self.hparams.eval_neg_sample_ratio,
            seed=self.hparams.seed,
        )
        print("Data ready.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
