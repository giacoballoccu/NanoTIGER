"""
demo_pipeline.py -- make sure trained artifacts exist, then the notebooks load them.

`ensure_trained()` is idempotent:

  * no embeddings yet -> synthesize the toy catalog and write pipeline artifacts
    (if you already have real data/item_emb.npy from the Gemma stage, it's kept)
  * no Semantic IDs / RQ-VAE checkpoint -> run the real rqvae.py training
  * no RecGPT checkpoint -> run the real train.py training

So the first notebook run trains the *full-config* models once (a minute or two)
and saves them; every run after that just reloads from disk.
"""

import sys
from pathlib import Path

sys.path.append("..")  # so we can import the repo's modules from notebooks/

from config import cfg, DATA, OUT  # noqa: E402
import toy_data  # noqa: E402


def _have(*paths):
    return all(Path(p).exists() for p in paths)


def ensure_trained(force=False, n_per_category=300):
    """Train (once) and return the paths the notebooks read."""
    emb_path = DATA / "item_emb.npy"
    items_path = DATA / "items.jsonl"
    seq_path = DATA / "sequences.json"

    if force or not emb_path.exists():
        print("[demo] writing toy catalog -> data/ ...")
        names, cats, brands, emb = toy_data.build_catalog(n_per_category, cfg.embed_dim, cfg.seed)
        seqs = toy_data.make_sequences(brands, seed=cfg.seed)   # brand-coherent histories
        toy_data.write_artifacts(DATA, names, cats, emb, seqs)
        print(f"[demo] {len(names)} items, {len(seqs)} user sequences")
    else:
        print(f"[demo] using existing embeddings at {emb_path}")

    if force or not _have(DATA / "semantic_ids.json", OUT / "rqvae.pt"):
        print("[demo] training RQ-VAE (rqvae.py) ...")
        import rqvae
        rqvae.main()
    else:
        print("[demo] RQ-VAE already trained")

    if force or not _have(OUT / "recgpt.pt"):
        print("[demo] training RecGPT (train.py) ...")
        import train
        train.main()
    else:
        print("[demo] RecGPT already trained")

    return {
        "items": items_path,
        "sequences": seq_path,
        "semantic_ids": DATA / "semantic_ids.json",
        "rqvae": OUT / "rqvae.pt",
        "recgpt": OUT / "recgpt.pt",
    }


if __name__ == "__main__":
    ensure_trained(force="--force" in sys.argv)
