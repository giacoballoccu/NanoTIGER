"""
embed_items.py -- STAGE 2: turn item text into dense semantic embeddings.

We use EmbeddingGemma (google/embeddinggemma-300m), Google's small open
embedding model. Two properties make it perfect here:

  * Matryoshka representation learning. A single forward pass yields a 768-d
    vector whose leading slices are *also* good embeddings. We ask for the
    smallest officially supported size, 128-d (`truncate_dim=128`), which keeps
    the downstream RQ-VAE tiny without meaningfully hurting quality.

  * It ships task-specific prompts. For indexing a corpus you embed with the
    "document" prompt; we use `encode_document`, which prepends the right
    instruction automatically.

Output:
    data/item_emb.npy   float32 array of shape (n_items, embed_dim), L2-normalized

Run:  python embed_items.py

Note: EmbeddingGemma is a gated model. The first run needs:
    pip install -U sentence-transformers
    huggingface-cli login          # after accepting the license on the model page
"""

import json

import numpy as np

from config import cfg, DATA
from common import get_device


def load_items():
    items = []
    with open(DATA / "items.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    items.sort(key=lambda r: r["item"])  # ensure row i == item id i
    return items


def main():
    items = load_items()
    texts = [r["text"] for r in items]
    print(f"[embed] {len(texts):,} items -> EmbeddingGemma @ {cfg.embed_dim}-d (Matryoshka)")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "This stage needs sentence-transformers:\n"
            "    pip install -U sentence-transformers\n"
            f"(import failed: {e})"
        )

    device = get_device()
    # truncate_dim activates the Matryoshka slice; the model still runs the full
    # network, we just keep the first `embed_dim` dimensions of the output.
    model = SentenceTransformer(
        cfg.embed_model,
        device=str(device),
        truncate_dim=cfg.embed_dim,
    )

    emb = model.encode_document(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors -> cosine geometry for RQ-VAE
        show_progress_bar=True,
    ).astype(np.float32)

    assert emb.shape == (len(texts), cfg.embed_dim), emb.shape
    np.save(DATA / "item_emb.npy", emb)
    print(f"[embed] saved data/item_emb.npy  shape={emb.shape}")
    print("Next: python rqvae.py")


if __name__ == "__main__":
    main()
