"""
config.py -- one place for every knob in the Semantic IDs pipeline.

Style note: like nanoGPT, we keep configuration boring and explicit. A single
small dataclass holds the handful of numbers that actually matter, and every
script imports the same `cfg`. No hidden globals, no YAML, no argparse soup.

The whole pipeline is four stages, each reading the previous stage's artifact:

    prepare_data.py  ->  data/items.jsonl, data/sequences.json
    embed_items.py   ->  data/item_emb.npy        (EmbeddingGemma, Matryoshka-128)
    rqvae.py         ->  data/semantic_ids.json    (RQ-VAE quantizer)
    train.py / eval  ->  out/recgpt.pt             (generative recommender)
"""

from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths. Everything generated lands in data/ and out/, both git-ignored.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


@dataclass
class Config:
    # ---- dataset ---------------------------------------------------------
    # Amazon Reviews 2023 (McAuley-Lab). Pick a *small* category so the whole
    # thing trains on a laptop. "All_Beauty" and "Musical_Instruments" are the
    # friendliest. The category name must match the HF config suffix.
    category: str = "Musical_Instruments"
    k_core: int = 5          # iteratively keep users/items with >= k_core interactions
    max_items: int = 8000    # cap the catalog so embedding + RQ-VAE stay quick
    max_seq_len: int = 20    # truncate each user history to the most recent N items
    min_seq_len: int = 5     # drop users with fewer than this many interactions
    seed: int = 1337         # the nanoGPT seed, for old times' sake

    # ---- embeddings ------------------------------------------------------
    # EmbeddingGemma is a Matryoshka model: one forward pass gives you a 768-d
    # vector whose *prefixes* are themselves valid embeddings. 128 is the
    # smallest officially supported truncation -- plenty for item content here,
    # and it keeps the RQ-VAE tiny.
    embed_model: str = "google/embeddinggemma-300m"
    embed_dim: int = 128

    # ---- RQ-VAE (turns a dense embedding into a tuple of discrete codes) --
    rq_levels: int = 3          # number of residual codebooks -> Semantic ID length
    rq_codebook_size: int = 256 # entries per codebook
    rq_latent_dim: int = 32     # encoder bottleneck
    rq_hidden: tuple = (256, 128)  # encoder/decoder MLP widths
    rq_beta: float = 0.25       # VQ commitment weight
    rq_epochs: int = 200
    rq_batch_size: int = 256
    rq_lr: float = 1e-3

    # ---- RecGPT (the generative recommender over Semantic ID tokens) -----
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    gpt_lr: float = 3e-4
    gpt_epochs: int = 30
    gpt_batch_size: int = 128
    weight_decay: float = 0.1

    # ---- evaluation ------------------------------------------------------
    eval_ks: tuple = (5, 10)
    beam_size: int = 30  # beam width for decoding next-item Semantic IDs

    @property
    def semantic_id_len(self) -> int:
        return self.rq_levels


cfg = Config()
