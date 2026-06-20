import pandas as pd


class PopularityRecommender:
    """Baseline: recommends globally most popular items unseen by the user."""

    def __init__(self) -> None:
        self._item_scores: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Rank items by number of positive interactions in train set."""
        positive_df = train_df[train_df["label"] == 1]
        self._item_scores = (
            positive_df.groupby("item_idx").size().sort_values(ascending=False)
        )
        return self

    def score(self, item_idx: int) -> float:
        """Return popularity score (interaction count); 0 if unseen in training."""
        if self._item_scores is None:
            raise RuntimeError("Call fit() before score()")
        return float(self._item_scores.get(item_idx, 0))

    def recommend(self, user_history: set[int], k: int = 10) -> list[int]:
        """Return top-K popular items not already seen by the user."""
        if self._item_scores is None:
            raise RuntimeError("Call fit() before recommend()")
        mask = ~self._item_scores.index.isin(user_history)
        return self._item_scores.index[mask][:k].tolist()

    def recommend_batch(
        self, user_histories: dict[int, set[int]], k: int = 10
    ) -> dict[int, list[int]]:
        """Return top-K recommendations for multiple users at once."""
        return {
            user_idx: self.recommend(history, k)
            for user_idx, history in user_histories.items()
        }
