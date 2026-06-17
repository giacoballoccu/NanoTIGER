# nanoTIGER

<img width="1365" height="716" alt="Gemini_Generated_Image_s1w5dis1w5dis1w5" src="https://github.com/user-attachments/assets/bfc4e706-cf11-4739-a4a0-300715500abd" />

The simplest, fastest repository for understanding **Semantic IDs** and
**generative recommendation**. In the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT): a handful of flat, hackable
files you can read in one sitting — no framework, no config hell.

nanoTIGER is a minimal, end-to-end reimplementation of
[**TIGER**](https://arxiv.org/abs/2305.05065) (*Recommender Systems with
Generative Retrieval*), wired up to Google's **EmbeddingGemma** and a
nanoGPT-style transformer:

```
prepare_data.py  ──▶  embed_items.py  ──▶  rqvae.py  ──▶  train.py  ──▶  eval.py
   item text          EmbeddingGemma      Semantic IDs     RecGPT       Recall@K
 user sequences      (Matryoshka-128)     (RQ-VAE codes)  (a tiny GPT)   NDCG@K
```

A **Semantic ID** replaces an item's arbitrary database id (`item #84321`) with
a short tuple of content-derived codes, e.g. `(37, 198, 4)`. Items that *mean*
similar things get codes that agree on their leading digits. Train an ordinary
GPT to **generate the next item's Semantic ID** from a user's history and you
have a recommender that's generative, content-aware, and great at cold start.

## install

```bash
pip install -r requirements.txt
```

Dependencies:

- [`pytorch`](https://pytorch.org) — the only thing the model code needs.
- [`numpy`](https://numpy.org)
- [`datasets`](https://github.com/huggingface/datasets) — for the Amazon Reviews data.
- [`sentence-transformers`](https://www.sbert.net) + [`transformers`](https://github.com/huggingface/transformers) — for EmbeddingGemma.

EmbeddingGemma is a **gated** model. Accept the license once and log in:

```bash
huggingface-cli login   # after accepting https://huggingface.co/google/embeddinggemma-300m
```

## quick start

Run the whole thing with one command:

```bash
bash run.sh
```

Or run the five stages yourself — each writes an artifact the next one reads:

```bash
python prepare_data.py   # 1. get dataset, extract item text  -> data/items.jsonl, data/sequences.json
python embed_items.py    # 2. text -> 128-d Matryoshka embeddings -> data/item_emb.npy
python rqvae.py          # 3. embeddings -> Semantic IDs       -> data/semantic_ids.json
python train.py          # 4. train RecGPT on ID sequences     -> out/recgpt.pt
python eval.py           # 5. Recall@K / NDCG@K vs popularity
```

Then see the payoff — items that share a Semantic ID prefix are semantically
related:

```bash
python show_neighbors.py
```

Or reproduce the full multi-domain results (codebook health, per-category
Recall/NDCG, joint-vs-specialized ablation) on the offline demo catalog:

```bash
python experiments.py
```

Everything is steered from a single small dataclass in `config.py`. Want
different catalogs? Edit `categories` (e.g. `["All_Beauty"]`, the smallest):

```bash
bash run.sh --categories All_Beauty
```

## I only have a macbook

You're fine — that's the point. nanoTIGER auto-selects the best device,
**CUDA → Apple MPS → CPU** (`common.get_device`), so the same code runs on your
Mac's GPU or a CUDA box with no changes. With the default `config.py` the whole
pipeline trains in minutes on an M-series laptop.

## how it works

**1. Content, not co-occurrence.** Each item is described by its text (title,
brand, categories, description). `embed_items.py` runs it through EmbeddingGemma,
keeping only the first **128** dimensions — EmbeddingGemma is a *Matryoshka*
model, so a prefix of the embedding is itself a valid embedding.

**2. RQ-VAE turns an embedding into a Semantic ID** (`rqvae.py`, the core idea).
A Residual-Quantized VAE encodes the embedding to a latent `z` and quantizes it
in `L` residual stages — each stage snaps the leftover to its nearest codebook
vector and records the index:

```
r₀ = z
cₗ  = argmin_k ‖ rₗ − Cₗ[k] ‖     ← the l-th Semantic ID digit
rₗ₊₁ = rₗ − Cₗ[cₗ]                ← residual passed to the next stage
```

The first code is coarse meaning, later codes refine it: a coarse-to-fine,
**prefix-shared** code space. Trained with the standard VQ-VAE reconstruction +
commitment loss and a straight-through estimator.

**3. RecGPT generates the next id** (`model.py` + `train.py`). User histories are
flattened into Semantic ID tokens (`tokenizer.py`) and a trimmed nanoGPT is
trained with the plain next-token objective. Because items *are* their Semantic
IDs, "predict the next token" *is* "predict the next item." At eval time
(`eval.py`) a constrained beam search generates the next item's digits and maps
them back to a real item.

## results

The repo ships a fully **offline, reproducible** demo so you can see everything
work without the gated EmbeddingGemma download: a synthetic **6-domain** catalog
(Musical Instruments, Video Games, Office Products, Home & Kitchen, Sports &
Outdoors, Pet Supplies) with brand-coherent user histories. One command trains
the codebook + recommender and prints every number below:

```bash
python experiments.py
```

Swap in real Amazon data by running `prepare_data.py` + `embed_items.py` first —
nothing else changes. *(The numbers below are from the synthetic demo, so read
them as evidence the machinery works and behaves sensibly, not as an Amazon
benchmark.)*

**The codebook is solid.** RQ-VAE on 3,600 items, held-out item split:

| reconstruction (train / val) | codebook used (levels 0/1/2) | unique IDs before disambiguation |
|---|---|---|
| 0.0001 / 0.0001 (no overfit) | 89% / 93% / 60% | 94.4% |

…and the top code is **pure per domain** — every first-code group holds items
from a single domain.

**Generative recommendation, per category** — temporal leave-one-out (predict
the last item; R@1 = next-item hit rate), 9,000 users, one joint model:

| Domain | R@1 | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| Musical Instruments | 0.010 | 0.080 | 0.155 | 0.042 | 0.066 |
| Video Games | 0.020 | 0.100 | 0.175 | 0.058 | 0.081 |
| Office Products | 0.015 | 0.065 | 0.135 | 0.042 | 0.063 |
| Home & Kitchen | 0.050 | 0.120 | 0.210 | 0.087 | 0.116 |
| Sports & Outdoors | 0.005 | 0.065 | 0.145 | 0.037 | 0.062 |
| Pet Supplies | 0.015 | 0.095 | 0.185 | 0.056 | 0.085 |
| **Overall** | **0.022** | **0.085** | **0.170** | **0.053** | **0.081** |
| Popularity baseline | 0.000 | 0.002 | 0.003 | 0.001 | 0.001 |

RecGPT beats popularity by ~50× — it generates the right content sub-cluster
from a user's history.

**Ablation — one joint model vs one model per domain** (same shared codebook).
A single joint RecGPT is competitive with, and often better than, specialized
per-domain models — positive transfer through the shared Semantic ID space:

| Domain | Joint R@10 | Specialized R@10 | Joint NDCG@10 | Specialized NDCG@10 |
|---|---|---|---|---|
| Musical Instruments | 0.155 | 0.145 | 0.066 | 0.069 |
| Video Games | 0.175 | 0.160 | 0.081 | 0.062 |
| Office Products | 0.135 | 0.175 | 0.063 | 0.070 |
| Home & Kitchen | 0.210 | 0.150 | 0.116 | 0.070 |
| Sports & Outdoors | 0.145 | 0.205 | 0.062 | 0.094 |
| Pet Supplies | 0.185 | 0.160 | 0.085 | 0.068 |

## notebooks

Two short, didactic notebooks (run in seconds, no gated model needed) that open
up the two core ideas:

- `notebooks/01_semantic_ids.ipynb` — train the RQ-VAE, watch the loss fall, and
  *see* that items sharing a first code occupy the same region of embedding space.
- `notebooks/02_recgpt.ipynb` — tokenize histories, train RecGPT, and generate the
  next item: a history inside one cluster yields next-item predictions in the
  same cluster.

## file guide

| file | what it is |
|------|------------|
| `config.py` | every hyperparameter, one small dataclass |
| `prepare_data.py` | stream Amazon reviews, k-core filter, build user sequences |
| `embed_items.py` | EmbeddingGemma, `truncate_dim=128` (Matryoshka) |
| `rqvae.py` | the Residual-Quantized VAE — **the core idea** |
| `tokenizer.py` | Semantic ID tuples ↔ transformer tokens |
| `model.py` | `RecGPT`, a trimmed nanoGPT |
| `train.py` | next-token training over Semantic ID sequences |
| `eval.py` | constrained beam search → per-split Recall@K / NDCG@K |
| `checkpoints.py` | one-line save / reload of the trained RQ-VAE and RecGPT |
| `experiments.py` | reproduce the results: codebook + per-category + ablation |
| `show_neighbors.py` | inspect items that share a Semantic ID prefix |
| `run.sh` | run all five stages in order |

## tweaking

All knobs live in `config.py`. The interesting ones:

- `categories` — list of Amazon Reviews 2023 categories to pool (default: 3).
- `k_core` — interaction-count threshold for users/items (default: 20).
- `embed_dim` — Matryoshka size (128 / 256 / 512 / 768).
- `rq_levels`, `rq_codebook_size` — Semantic ID length and resolution.
- `n_layer`, `n_embd` — scale RecGPT up for bigger catalogs.

Evaluation uses a **temporal split** (last item = test, second-to-last = val);
the RQ-VAE keeps a held-out item split and reports codebook utilization so you
can tell a healthy codebook from a collapsed one.

## todos

Done so far: loss curves + cluster visualization (notebooks), multi-domain
catalog, temporal split, per-category Recall/NDCG, dead-code revival, joint-vs-
specialized ablation. Coming next:

- [ ] Run the real EmbeddingGemma + Amazon pipeline end-to-end and report numbers.
- [ ] Add a SASRec / item-id GPT baseline next to the Semantic ID model.
- [ ] Trie-constrained beam search (only ever decode valid item IDs).

## acknowledgements

- Rajput et al., *Recommender Systems with Generative Retrieval* (**TIGER**), 2023.
- Lee et al., *Autoregressive Image Generation using Residual Quantization* (**RQ-VAE**), 2022.
- van den Oord et al., *Neural Discrete Representation Learning* (**VQ-VAE**), 2017.
- Google DeepMind, **EmbeddingGemma**, 2025.
- Andrej Karpathy, [**nanoGPT**](https://github.com/karpathy/nanoGPT) — the inspiration for this repo's style.
