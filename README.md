# nanoTIGER

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

Everything is steered from a single small dataclass in `config.py`. Want a
different catalog? Change one line (`All_Beauty` is the smallest Amazon category):

```bash
bash run.sh --category All_Beauty
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

## notebooks

Two short, didactic notebooks (synthetic data, runs in seconds, no gated model
needed) that open up the two core ideas:

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
| `eval.py` | constrained beam search → Recall@K / NDCG@K |
| `show_neighbors.py` | inspect items that share a Semantic ID prefix |
| `run.sh` | run all five stages in order |

## tweaking

All knobs live in `config.py`. The interesting ones:

- `category` — any Amazon Reviews 2023 category.
- `embed_dim` — Matryoshka size (128 / 256 / 512 / 768).
- `rq_levels`, `rq_codebook_size` — Semantic ID length and resolution.
- `n_layer`, `n_embd` — scale RecGPT up for bigger catalogs.

## todos

This is the **base version**. Coming next:

- [ ] Plots: RQ-VAE reconstruction/commitment loss curves and RecGPT training loss.
- [ ] Cluster visualization — show that prefix-shared Semantic IDs form coherent groups.
- [ ] More datasets (multiple Amazon categories + MovieLens) and a results table.

## acknowledgements

- Rajput et al., *Recommender Systems with Generative Retrieval* (**TIGER**), 2023.
- Lee et al., *Autoregressive Image Generation using Residual Quantization* (**RQ-VAE**), 2022.
- van den Oord et al., *Neural Discrete Representation Learning* (**VQ-VAE**), 2017.
- Google DeepMind, **EmbeddingGemma**, 2025.
- Andrej Karpathy, [**nanoGPT**](https://github.com/karpathy/nanoGPT) — the inspiration for this repo's style.
