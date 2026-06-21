import json
import subprocess
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import mlflow
import mlflow.onnx
import onnx as onnx_lib
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import DictConfig
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

from data.dataset import RecsDataModule
from data.download import download_data, pull_dvc
from models.ncf import NCFModel
from models.popularity import PopularityRecommender


def evaluate_popularity(
    baseline: PopularityRecommender,
    dm: RecsDataModule,
    top_k: int,
    stage: str,
) -> dict[str, float]:
    metrics = MetricCollection(
        {
            f"precision_at_{top_k}": RetrievalPrecision(top_k=top_k),
            f"recall_at_{top_k}": RetrievalRecall(top_k=top_k),
            f"ndcg_at_{top_k}": RetrievalNormalizedDCG(top_k=top_k),
        }
    )
    loader = dm.val_dataloader() if stage == "val" else dm.test_dataloader()
    for user_ids, item_ids, labels in loader:
        scores = torch.tensor(
            [baseline.score(int(iid)) for iid in item_ids],
            dtype=torch.float,
        )
        metrics.update(scores, labels.long(), indexes=user_ids)
    return {f"{stage}/{k}": float(v) for k, v in metrics.compute().items()}


def run_popularity_baseline(dm: RecsDataModule, cfg: DictConfig) -> None:
    baseline = PopularityRecommender()
    baseline.fit(dm.train_df)

    val_metrics = evaluate_popularity(baseline, dm, cfg.model.top_k, "val")
    test_metrics = evaluate_popularity(baseline, dm, cfg.model.top_k, "test")
    all_metrics = {**val_metrics, **test_metrics}

    mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.training.mlflow_experiment_name)
    with mlflow.start_run(
        run_name="popularity-baseline", tags={"git_commit": get_git_commit()}
    ):
        mlflow.log_metrics(all_metrics)

    print("Popularity baseline:", all_metrics)


def build_datamodule(cfg: DictConfig) -> RecsDataModule:
    return RecsDataModule(
        data_dir=Path(cfg.data.data_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        neg_sample_ratio=cfg.data.neg_sample_ratio,
        eval_neg_sample_ratio=cfg.data.eval_neg_sample_ratio,
        seed=cfg.data.seed,
        min_positive_rating=cfg.data.min_positive_rating,
    )


def build_model(cfg: DictConfig, num_users: int, num_items: int) -> NCFModel:
    return NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg.model.embedding_dim,
        hidden_layers=list(cfg.model.hidden_layers),
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        top_k=cfg.model.top_k,
    )


def build_trainer(cfg: DictConfig, checkpoint_dir: Path) -> L.Trainer:
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss", patience=cfg.training.early_stopping_patience
        ),
    ]
    return L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        fast_dev_run=cfg.training.fast_dev_run,
        logger=build_loggers(cfg),
    )


def save_plots(trainer: L.Trainer, plots_dir: Path, top_k: int = 10) -> None:
    csv_logger = next((lg for lg in trainer.loggers if isinstance(lg, CSVLogger)), None)
    if csv_logger is None:
        return

    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if not metrics_path.exists():
        return

    df = pd.read_csv(metrics_path)
    if df.empty or "epoch" not in df.columns:
        return

    plots_dir.mkdir(exist_ok=True)

    def plot_columns(columns: dict[str, str], filename: str, title: str) -> None:
        fig, ax = plt.subplots()
        for col, label in columns.items():
            if col in df.columns:
                subset = df[["epoch", col]].dropna()
                ax.plot(subset["epoch"], subset[col], label=label)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plots_dir / filename}")

    plot_columns({"train/loss": "train"}, "loss_train.png", "Train Loss")
    plot_columns({"val/loss": "val"}, "loss_val.png", "Val Loss")
    plot_columns(
        {f"val/precision@{top_k}": "val"},
        f"precision_at_{top_k}.png",
        f"Precision@{top_k}",
    )
    plot_columns(
        {f"val/recall@{top_k}": "val"}, f"recall_at_{top_k}.png", f"Recall@{top_k}"
    )
    plot_columns({f"val/ndcg@{top_k}": "val"}, f"ndcg_at_{top_k}.png", f"NDCG@{top_k}")


def export_onnx(model: NCFModel, path: Path) -> None:
    model.eval()
    cpu_model = model.cpu()
    dummy_users = torch.zeros(1, dtype=torch.long)
    dummy_items = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(
        cpu_model,
        (dummy_users, dummy_items),
        path,
        input_names=["user_ids", "item_ids"],
        output_names=["scores"],
        dynamic_axes={
            "user_ids": {0: "batch_size"},
            "item_ids": {0: "batch_size"},
            "scores": {0: "batch_size"},
        },
        opset_version=17,
    )
    meta = {"num_users": model.hparams.num_users, "num_items": model.hparams.num_items}
    with path.with_suffix(".json").open("w") as f:
        json.dump(meta, f)
    print(f"ONNX model saved to {path}")


def log_onnx_to_mlflow(onnx_path: Path, cfg: DictConfig) -> None:
    onnx_model = onnx_lib.load(str(onnx_path))
    mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.training.mlflow_experiment_name)
    with mlflow.start_run(run_name="ncf-onnx", tags={"git_commit": get_git_commit()}):
        mlflow.onnx.log_model(onnx_model, "model")
    print("ONNX model logged to MLflow")


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def build_loggers(cfg: DictConfig) -> list:
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.training.mlflow_experiment_name,
        tracking_uri=cfg.training.mlflow_tracking_uri,
        tags={"git_commit": get_git_commit()},
    )
    csv_logger = CSVLogger(save_dir=".", name="logs")
    return [mlf_logger, csv_logger]


def run_training(cfg: DictConfig) -> None:
    if cfg.training.download:
        download_data(Path(cfg.data.data_dir))
    if cfg.training.pull:
        pull_dvc()

    try:
        cwd = Path(get_original_cwd())
    except Exception:
        cwd = Path.cwd()

    dm = build_datamodule(cfg)
    dm.setup("fit")

    model = build_model(cfg, num_users=dm.num_users, num_items=dm.num_items)
    checkpoint_dir = cwd / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    trainer = build_trainer(cfg, checkpoint_dir=checkpoint_dir)
    trainer.fit(model, datamodule=dm)

    save_plots(trainer, cwd / cfg.training.plots_dir, top_k=cfg.model.top_k)

    onnx_path = cwd / cfg.training.onnx_path
    onnx_path.parent.mkdir(exist_ok=True)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_model = NCFModel.load_from_checkpoint(best_ckpt) if best_ckpt else model
    export_onnx(best_model, onnx_path)

    if not cfg.training.fast_dev_run:
        log_onnx_to_mlflow(onnx_path, cfg)
        run_popularity_baseline(dm, cfg)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
