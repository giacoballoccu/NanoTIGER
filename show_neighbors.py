"""
show_neighbors.py -- the payoff: items that share a Semantic ID prefix.

The whole promise of Semantic IDs is that *meaning* is baked into the id: items
with similar content get codes that agree on their leading digits. This script
makes that visible. Pick a few items and print the other items sharing their
first one or two RQ-VAE codes -- you should see semantically related products.

Run:  python show_neighbors.py
      python show_neighbors.py --n 8 --prefix 2
"""

import argparse
import json
from collections import defaultdict

from config import cfg, DATA
from common import seed_everything, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="how many anchor items to show")
    ap.add_argument("--prefix", type=int, default=1, help="prefix length to group by")
    args = ap.parse_args()
    seed_everything(cfg.seed)

    texts = {}
    with open(DATA / "items.jsonl") as f:
        for line in f:
            r = json.loads(line)
            texts[r["item"]] = r["text"]

    ids = load_json(DATA / "semantic_ids.json")["ids"]
    prefix = args.prefix

    # group items by their leading `prefix` codes
    groups = defaultdict(list)
    for item, code in enumerate(ids):
        groups[tuple(code[:prefix])].append(item)

    # show the largest groups -- those are the clearest clusters
    shown = 0
    for key, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(members) < 2:
            continue
        print(f"\n=== Semantic ID prefix {key}  ({len(members)} items) ===")
        for item in members[:6]:
            full = ids[item]
            short = texts[item][:80].replace("\n", " ")
            print(f"  {tuple(full)}  {short}")
        shown += 1
        if shown >= args.n:
            break


if __name__ == "__main__":
    main()
