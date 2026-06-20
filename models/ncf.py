import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)


class NCFModel(L.LightningModule):
    """Neural Collaborative Filtering for implicit feedback."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        top_k: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        mlp_hidden = hidden_layers if hidden_layers is not None else [64, 32, 16]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layer_sizes = [embedding_dim * 2] + mlp_hidden
        layers: list[nn.Module] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(layer_sizes[-1], 1), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

        self._loss_fn = nn.BCELoss()

        # Ranking metrics — torchmetrics groupes by user_ids automatically
        ranking_metrics = MetricCollection(
            {
                f"precision@{top_k}": RetrievalPrecision(top_k=top_k),
                f"recall@{top_k}": RetrievalRecall(top_k=top_k),
                f"ndcg@{top_k}": RetrievalNormalizedDCG(top_k=top_k),
            }
        )
        self._val_metrics = ranking_metrics.clone(prefix="val/")
        self._test_metrics = ranking_metrics.clone(prefix="test/")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(concat).squeeze(-1)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        user_ids, item_ids, labels = batch
        predictions = self(user_ids, item_ids)
        loss = self._loss_fn(predictions, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _eval_step(
        self, batch: tuple, metrics: MetricCollection, stage: str
    ) -> torch.Tensor:
        user_ids, item_ids, labels = batch
        predictions = self(user_ids, item_ids)
        loss = self._loss_fn(predictions, labels)

        metrics.update(predictions, labels.long(), indexes=user_ids)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, self._val_metrics, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, self._test_metrics, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
