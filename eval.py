"""
eval.py -- STAGE 5: evaluate next-item recommendation.

For each user we feed the history (minus the held-out last item) and ask RecGPT
to *generate* the next item's Semantic ID, digit by digit, with a constrained
beam search: at digit position p the beam may only emit tokens from position p's
slice of the vocabulary. Completed beams are mapped back to real items via the
Semantic ID -> item table, items already in the history are removed, and the
top-K survivors are scored against the true held-out item.

Metrics: Recall@K and NDCG@K (the usual leave-one-out sequential-rec metrics).
A popularity baseline is printed alongside for context.

Run:  python eval.py
"""

import math
from collections import Counter

import torch
import torch.nn.functional as F

from config import cfg, DATA, OUT
from common import get_device, seed_everything, load_json
from tokenizer import build_tokenizer
from model import GPTConfig, RecGPT


@torch.no_grad()
def beam_search_items(model, tok, history, device, beam_size, max_items_out):
    """Generate next-item Semantic IDs by constrained beam search.
    Returns an ordered list of candidate item ids (best first)."""
    ctx = tok.encode_sequence(history)              # BOS + history tokens
    base = torch.tensor(ctx, dtype=torch.long, device=device)
    beams = [(0.0, [])]                              # (logprob, generated tokens)
    for p in range(tok.D):
        lo, hi = tok.valid_token_range(p)
        cand = []
        # batch all beams through the model at once
        seqs = torch.stack([torch.cat([base, torch.tensor(g, dtype=torch.long, device=device)])
                            for _, g in beams])
        logits, _ = model(seqs)
        logp = F.log_softmax(logits[:, -1, lo:hi], dim=-1)  # only this position's codes
        for b, (score, gen) in enumerate(beams):
            topv, topi = logp[b].topk(min(beam_size, hi - lo))
            for v, i in zip(topv.tolist(), topi.tolist()):
                cand.append((score + v, gen + [lo + i]))
        cand.sort(key=lambda x: -x[0])
        beams = cand[:beam_size]

    items, seen = [], set()
    for _, gen in beams:
        item = tok.tokens_to_item(gen)
        if item is not None and item not in seen:
            seen.add(item)
            items.append(item)
        if len(items) >= max_items_out:
            break
    return items


def recall_ndcg(ranked, target, k):
    topk = ranked[:k]
    if target in topk:
        rank = topk.index(target)
        return 1.0, 1.0 / math.log2(rank + 2)
    return 0.0, 0.0


def main():
    seed_everything(cfg.seed)
    device = get_device()

    sequences = load_json(DATA / "sequences.json")["train"]
    sem = load_json(DATA / "semantic_ids.json")
    tok = build_tokenizer(sem["ids"])

    ckpt = torch.load(OUT / "recgpt.pt", map_location=device, weights_only=False)
    gptcfg = GPTConfig(**ckpt["gptcfg"])
    model = RecGPT(gptcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # popularity baseline: rank items by training-history frequency
    pop = Counter(i for s in sequences for i in s[:-1])
    pop_ranked = [i for i, _ in pop.most_common()]

    ks = cfg.eval_ks
    maxk = max(ks)
    agg = {k: [0.0, 0.0] for k in ks}            # RecGPT: [recall_sum, ndcg_sum]
    pop_agg = {k: [0.0, 0.0] for k in ks}
    n_eval = 0

    eval_users = [s for s in sequences if len(s) >= 2]
    print(f"[eval] scoring {len(eval_users):,} users (leave-one-out) ...")
    for n, seq in enumerate(eval_users):
        history, target = seq[:-1], seq[-1]
        ranked = beam_search_items(model, tok, history, device, cfg.beam_size, maxk)
        # baseline ranking with seen items removed
        seen = set(history)
        pop_rank = [i for i in pop_ranked if i not in seen]
        for k in ks:
            r, g = recall_ndcg(ranked, target, k)
            agg[k][0] += r
            agg[k][1] += g
            pr, pg = recall_ndcg(pop_rank, target, k)
            pop_agg[k][0] += pr
            pop_agg[k][1] += pg
        n_eval += 1
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(eval_users)} users ...")

    print(f"\n=== Results over {n_eval:,} users ===")
    print(f"{'metric':<12}{'RecGPT':>12}{'Popularity':>14}")
    for k in ks:
        print(f"Recall@{k:<6}{agg[k][0]/n_eval:>12.4f}{pop_agg[k][0]/n_eval:>14.4f}")
        print(f"NDCG@{k:<8}{agg[k][1]/n_eval:>12.4f}{pop_agg[k][1]/n_eval:>14.4f}")


if __name__ == "__main__":
    main()
