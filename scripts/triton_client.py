"""Test client for NCF model served via Triton Inference Server.

Install client library: pip install tritonclient[http]

Usage:
    python scripts/triton_client.py --user_id=42
    python scripts/triton_client.py --user_id=42 --num_items=197747 --top_k=10
"""

import json
from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient


def query_triton(
    user_id: int,
    num_items: int = 0,
    top_k: int = 10,
    model_name: str = "ncf",
    url: str = "localhost:8000",
    meta_path: str = "models/ncf.json",
) -> list[int]:
    """Send inference request to Triton and return top-K item indices.

    Args:
        user_id: encoded user index (0-based, from training)
        num_items: total number of items; if 0, reads from meta_path
        top_k: number of recommendations to return
        model_name: Triton model name (must match triton/ncf/config.pbtxt)
        url: Triton HTTP endpoint
        meta_path: path to ncf.json with num_items metadata
    """
    if num_items == 0:
        with Path(meta_path).open() as f:
            meta = json.load(f)
        num_items = meta["num_items"]

    user_ids = np.full(num_items, user_id, dtype=np.int64)
    item_ids = np.arange(num_items, dtype=np.int64)

    client = httpclient.InferenceServerClient(url=url)

    inputs = [
        httpclient.InferInput("user_ids", user_ids.shape, "INT64"),
        httpclient.InferInput("item_ids", item_ids.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(user_ids)
    inputs[1].set_data_from_numpy(item_ids)

    outputs = [httpclient.InferRequestedOutput("scores")]

    response = client.infer(model_name, inputs=inputs, outputs=outputs)
    scores: np.ndarray = response.as_numpy("scores")

    top_items: list[int] = np.argsort(scores)[::-1][:top_k].tolist()
    print(f"Top-{top_k} recommendations for user_id={user_id} (via Triton):")
    for rank, item_idx in enumerate(top_items, 1):
        print(f"  {rank:2d}. item_idx={item_idx}  score={scores[item_idx]:.4f}")

    return top_items


if __name__ == "__main__":
    fire.Fire(query_triton)
