from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from infer import recommend


def train(**overrides: str) -> None:
    """Train the NCF model. Accepts Hydra overrides as keyword arguments.

    Example:
        python commands.py train model.lr=0.001 training.max_epochs=5
    """
    GlobalHydra.instance().clear()
    configs_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        override_list = [f"{k}={v}" for k, v in overrides.items()]
        cfg = compose(config_name="config", overrides=override_list)
        from scripts.train import run_training

        run_training(cfg)


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": recommend})
