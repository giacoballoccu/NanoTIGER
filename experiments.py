"""
experiments.py -- reproduce the numbers in the README.

Runs the whole thing end-to-end on the offline multi-domain demo catalog
(no gated model required):

  1. build a scaled, multi-domain catalog and write pipeline artifacts
  2. train the RQ-VAE  -> a shared codebook + Semantic IDs (reports health)
  3. train one joint RecGPT on ALL domains
  4. evaluate per category: Recall@k / NDCG@k for k in {1,5,10}  (k=1 = next-item)
  5. ablation: a specialized RecGPT trained on a single domain, evaluated on that
     same domain, vs the joint model

Everything is sampled where it would otherwise be slow. Run:  python experiments.py
"""

import sys
from collections import Counter, defaultdict

sys.path.append("notebooks")  # for toy_data

import numpy as np

from config import cfg, DATA
from common import get_device, seed_everything, load_json
from tokenizer import build_tokenizer
import toy_data
import rqvae
from train import train_recgpt
from eval import evaluate

KS = (1, 5, 10)             # k=1 is "next-item" (HR@1)
N_PER_CAT = 600             # items per domain  -> 6 domains * 600 = 3,600 items
N_USERS = 9000              # synthetic user histories
EVAL_USERS_PER_DOMAIN = 200  # sampled per domain to keep beam search quick


def pop_rank_of(sequences):
    return [i for i, _ in Counter(i for s in sequences for i in s[:-2]).most_common()]


def fmt_row(label, m):
    return (f"| {label:<22} | {m[1]['recall']:.3f} | {m[5]['recall']:.3f} | "
            f"{m[10]['recall']:.3f} | {m[5]['ndcg']:.3f} | {m[10]['ndcg']:.3f} |")


def main():
    seed_everything(cfg.seed)
    device = get_device()

    # 1. data ---------------------------------------------------------------
    names, cats, brands, emb = toy_data.build_catalog(N_PER_CAT, cfg.embed_dim, cfg.seed)
    seqs = toy_data.make_sequences(brands, n_users=N_USERS, seed=cfg.seed)
    toy_data.write_artifacts(DATA, names, cats, emb, seqs)
    doms = toy_data.domains()
    print(f"[exp] {len(names):,} items across {len(doms)} domains | {len(seqs):,} users\n")

    # 2. RQ-VAE codebook ----------------------------------------------------
    rqvae.main()
    tok = build_tokenizer(load_json(DATA / "semantic_ids.json")["ids"])

    # 3. joint RecGPT -------------------------------------------------------
    print("\n[exp] training JOINT RecGPT on all domains ...")
    joint, joint_cfg = train_recgpt(seqs, tok, device)
    from checkpoints import save_recgpt
    from config import OUT
    save_recgpt(joint, joint_cfg, OUT / "recgpt.pt")   # keep artifacts consistent

    # group eval users by domain (each user is brand-coherent -> single domain)
    by_dom = defaultdict(list)
    for s in seqs:
        by_dom[cats[s[0]]].append(s)
    rng = np.random.default_rng(cfg.seed)
    eval_users = {d: [by_dom[d][i] for i in
                      rng.choice(len(by_dom[d]),
                                 min(EVAL_USERS_PER_DOMAIN, len(by_dom[d])), replace=False)]
                  for d in doms}

    # 4. per-category evaluation of the joint model -------------------------
    print("\n[exp] evaluating joint model per category ...")
    joint_rows, all_users = {}, []
    for d in doms:
        pr = pop_rank_of(by_dom[d])                       # per-domain popularity
        joint_rows[d] = evaluate(joint, tok, eval_users[d], pr, device, ks=KS)
        all_users += eval_users[d]
    overall = evaluate(joint, tok, all_users, pop_rank_of(seqs), device, ks=KS)

    # 5. ablation: specialized (single-domain) vs joint ---------------------
    print("\n[exp] ablation: training a specialized RecGPT per domain ...")
    abl = {}
    for d in doms:
        spec, _ = train_recgpt(by_dom[d], tok, device, verbose=False)
        pr = pop_rank_of(by_dom[d])
        abl[d] = evaluate(spec, tok, eval_users[d], pr, device, ks=KS)
        print(f"   {d:<22} specialized R@10={abl[d][10]['recall']:.3f} "
              f"| joint R@10={joint_rows[d][10]['recall']:.3f}")

    # ---- markdown tables --------------------------------------------------
    print("\n\n### Per-category results (joint model, temporal test split)\n")
    print("| Domain | R@1 | R@5 | R@10 | NDCG@5 | NDCG@10 |")
    print("|---|---|---|---|---|---|")
    for d in doms:
        print(fmt_row(d.replace("_", " "), joint_rows[d]))
    print(fmt_row("**Overall**", overall))
    pop = {k: {"recall": overall[k]["pop_recall"], "ndcg": overall[k]["pop_ndcg"]} for k in KS}
    print(fmt_row("Popularity baseline", pop))

    print("\n\n### Ablation: joint vs specialized (R@10 / NDCG@10, test)\n")
    print("| Domain | Joint R@10 | Specialized R@10 | Joint NDCG@10 | Specialized NDCG@10 |")
    print("|---|---|---|---|---|")
    for d in doms:
        print(f"| {d.replace('_', ' '):<20} | {joint_rows[d][10]['recall']:.3f} | "
              f"{abl[d][10]['recall']:.3f} | {joint_rows[d][10]['ndcg']:.3f} | "
              f"{abl[d][10]['ndcg']:.3f} |")


if __name__ == "__main__":
    main()
