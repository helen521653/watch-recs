import json
from pathlib import Path

import fire
import numpy as np
import onnxruntime as ort


def recommend(
    user_id: int,
    onnx_path: str = "models/ncf.onnx",
    top_k: int = 10,
) -> list[int]:
    """Return top-K item indices for a given user.

    Args:
        user_id: encoded user index (integer, 0-based from training)
        onnx_path: path to the exported ONNX model file
        top_k: number of recommendations to return
    """
    onnx_path = Path(onnx_path)
    meta_path = onnx_path.with_suffix(".json")
    with meta_path.open() as f:
        meta = json.load(f)
    num_items: int = meta["num_items"]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    user_ids = np.full(num_items, user_id, dtype=np.int64)
    item_ids = np.arange(num_items, dtype=np.int64)
    scores: np.ndarray = sess.run(
        ["scores"], {"user_ids": user_ids, "item_ids": item_ids}
    )[0]

    top_items: list[int] = np.argsort(scores)[::-1][:top_k].tolist()
    print(f"Top-{top_k} recommendations for user_id={user_id}:")
    for rank, item_idx in enumerate(top_items, 1):
        print(f"  {rank:2d}. item_idx={item_idx}  score={scores[item_idx]:.4f}")
    return top_items


if __name__ == "__main__":
    fire.Fire(recommend)
