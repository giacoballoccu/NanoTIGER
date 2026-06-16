"""
prepare_data.py -- STAGE 1: get the dataset and extract item text.

We use the Amazon Reviews 2023 dataset (McAuley-Lab), the standard benchmark in
the Semantic IDs / generative-retrieval literature (this is the dataset family
the TIGER paper uses). Each product has rich text -- title, features,
description, categories -- which is exactly what we want to turn into a
*semantic* identifier.

What this script produces:

    data/items.jsonl    one line per item: {"item": int, "asin": str, "text": str}
    data/sequences.json {"train": [[item ids...], ...], "user_ids": [...]}
    data/meta.json      bookkeeping (n_items, n_users, category, ...)

The recipe is the textbook sequential-recommendation pipeline:
    1. stream raw reviews  (user, item, timestamp)
    2. iterative k-core filtering: keep only users/items with >= k interactions
    3. cap the catalog size so the rest of the pipeline stays laptop-friendly
    4. pull item metadata text for the surviving items
    5. order each user's interactions by time -> one sequence per user

Run:  python prepare_data.py            # uses category in config.py
      python prepare_data.py --category All_Beauty --max-reviews 150000
"""

import argparse
import json
from collections import defaultdict

from config import cfg, DATA
from common import save_json, seed_everything

# Streaming reviews keeps memory + download bounded even for big categories.
# We stop after this many raw interactions, then filter down from there.
DEFAULT_MAX_REVIEWS = 200_000


def _load_stream(config_name: str, split: str = "full"):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "This stage needs the `datasets` library:\n"
            "    pip install datasets\n"
            f"(import failed: {e})"
        )
    return load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        split=split,
        trust_remote_code=True,
        streaming=True,
    )


def stream_interactions(category: str, max_reviews: int):
    """Yield (user_id, parent_asin, timestamp) triples from streamed reviews."""
    ds = _load_stream(f"raw_review_{category}")
    n = 0
    for row in ds:
        user, item, ts = row.get("user_id"), row.get("parent_asin"), row.get("timestamp")
        if not user or not item or ts is None:
            continue
        yield user, item, ts
        n += 1
        if n >= max_reviews:
            break


def k_core_filter(interactions, k_core, min_seq, max_items):
    """Iterative k-core: repeatedly drop users/items below the interaction
    threshold until the graph is stable. Then cap the catalog to `max_items`
    by popularity (and re-run the filter so nothing falls below k afterwards).
    """
    # de-duplicate (user, item): keep earliest timestamp for that pair
    pair_ts = {}
    for u, i, ts in interactions:
        key = (u, i)
        if key not in pair_ts or ts < pair_ts[key]:
            pair_ts[key] = ts
    edges = [(u, i, ts) for (u, i), ts in pair_ts.items()]

    def counts(edges):
        uc, ic = defaultdict(int), defaultdict(int)
        for u, i, _ in edges:
            uc[u] += 1
            ic[i] += 1
        return uc, ic

    # cap catalog to the most popular items first
    _, ic = counts(edges)
    if len(ic) > max_items:
        keep_items = {i for i, _ in sorted(ic.items(), key=lambda x: -x[1])[:max_items]}
        edges = [(u, i, ts) for u, i, ts in edges if i in keep_items]

    # iterate to a stable k-core
    while True:
        uc, ic = counts(edges)
        before = len(edges)
        edges = [
            (u, i, ts) for u, i, ts in edges
            if uc[u] >= min_seq and ic[i] >= k_core
        ]
        if len(edges) == before:
            break
    return edges


def build_sequences(edges, max_seq_len):
    """Group edges by user, order by timestamp, remap to dense integer ids."""
    by_user = defaultdict(list)
    for u, i, ts in edges:
        by_user[u].append((ts, i))

    item2id, sequences, user_ids = {}, [], []
    for u in sorted(by_user):
        ordered = [i for _, i in sorted(by_user[u])][-max_seq_len:]
        seq = []
        for asin in ordered:
            if asin not in item2id:
                item2id[asin] = len(item2id)
            seq.append(item2id[asin])
        sequences.append(seq)
        user_ids.append(u)
    return sequences, user_ids, item2id


def fetch_item_text(category, asins):
    """Stream item metadata and assemble one text blob per kept item."""
    want = set(asins)
    text = {}
    ds = _load_stream(f"raw_meta_{category}")
    for row in ds:
        asin = row.get("parent_asin")
        if asin not in want or asin in text:
            continue
        parts = []
        if row.get("title"):
            parts.append(str(row["title"]))
        if row.get("store"):
            parts.append(f"Brand: {row['store']}")
        cats = row.get("categories") or []
        if cats:
            parts.append("Categories: " + " > ".join(map(str, cats)))
        feats = row.get("features") or []
        if feats:
            parts.append(" ".join(map(str, feats)))
        desc = row.get("description") or []
        if desc:
            parts.append(" ".join(map(str, desc)))
        text[asin] = " | ".join(parts).strip()
        if len(text) == len(want):
            break
    # any item without metadata falls back to its asin so nothing is empty
    for a in want:
        text.setdefault(a, a)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=cfg.category)
    ap.add_argument("--max-reviews", type=int, default=DEFAULT_MAX_REVIEWS)
    args = ap.parse_args()

    seed_everything(cfg.seed)
    print(f"[1/5] streaming up to {args.max_reviews:,} reviews from '{args.category}' ...")
    interactions = list(stream_interactions(args.category, args.max_reviews))
    print(f"      got {len(interactions):,} raw interactions")

    print(f"[2/5] {cfg.k_core}-core filtering (min user history {cfg.min_seq_len}) ...")
    edges = k_core_filter(interactions, cfg.k_core, cfg.min_seq_len, cfg.max_items)
    print(f"      {len(edges):,} interactions survive")

    print("[3/5] building per-user sequences ...")
    sequences, user_ids, item2id = build_sequences(edges, cfg.max_seq_len)
    id2asin = {v: k for k, v in item2id.items()}
    print(f"      {len(sequences):,} users, {len(item2id):,} items")

    print("[4/5] fetching item metadata text ...")
    text_by_asin = fetch_item_text(args.category, list(item2id.keys()))

    print("[5/5] writing artifacts ...")
    with open(DATA / "items.jsonl", "w") as f:
        for iid in range(len(item2id)):
            asin = id2asin[iid]
            rec = {"item": iid, "asin": asin, "text": text_by_asin.get(asin, asin)}
            f.write(json.dumps(rec) + "\n")
    save_json({"train": sequences, "user_ids": user_ids}, DATA / "sequences.json")
    save_json(
        {
            "n_items": len(item2id),
            "n_users": len(sequences),
            "category": args.category,
            "k_core": cfg.k_core,
        },
        DATA / "meta.json",
    )
    print(f"\nDone. {len(item2id):,} items -> data/items.jsonl")
    print("Next: python embed_items.py")


if __name__ == "__main__":
    main()
