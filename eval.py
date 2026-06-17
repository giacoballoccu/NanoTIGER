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


def evaluate(model, tok, sequences, pop_ranked, device, ks=None, beam=None,
             split="test"):
    """Average Recall@k and NDCG@k (and a popularity baseline) over users on a
    temporal split. split='test' -> predict seq[-1] from seq[:-1];
    split='val' -> predict seq[-2] from seq[:-2]. Returns dict:
        {k: {'recall':_, 'ndcg':_, 'pop_recall':_, 'pop_ndcg':_}}.
    """
    ks = ks or cfg.eval_ks
    beam = beam or cfg.beam_size
    cut, tgt = (-1, -1) if split == "test" else (-2, -2)
    agg = {k: [0.0, 0.0, 0.0, 0.0] for k in ks}
    users = [s for s in sequences if len(s) >= 3]
    for seq in users:
        history, target = seq[:cut], seq[tgt]
        ranked = beam_search_items(model, tok, history, device, beam, max(ks))
        pop_rank = [i for i in pop_ranked if i not in set(history)]
        for k in ks:
            r, g = recall_ndcg(ranked, target, k)
            pr, pg = recall_ndcg(pop_rank, target, k)
            agg[k][0] += r; agg[k][1] += g; agg[k][2] += pr; agg[k][3] += pg
    n = max(1, len(users))
    return {k: {"recall": v[0] / n, "ndcg": v[1] / n,
                "pop_recall": v[2] / n, "pop_ndcg": v[3] / n} for k, v in agg.items()}


def main():
    seed_everything(cfg.seed)
    device = get_device()

    sequences = load_json(DATA / "sequences.json")["train"]
    tok = build_tokenizer(load_json(DATA / "semantic_ids.json")["ids"])

    ckpt = torch.load(OUT / "recgpt.pt", map_location=device, weights_only=False)
    model = RecGPT(GPTConfig(**ckpt["gptcfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # popularity baseline ranks items by frequency in the training portion only
    pop_ranked = [i for i, _ in Counter(i for s in sequences for i in s[:-2]).most_common()]

    print(f"[eval] scoring {sum(len(s) >= 3 for s in sequences):,} users (temporal split) ...")
    val = evaluate(model, tok, sequences, pop_ranked, device, split="val")
    test = evaluate(model, tok, sequences, pop_ranked, device, split="test")

    print(f"\n=== Results (temporal split) ===")
    print(f"{'metric':<12}{'val':>10}{'test':>10}{'test (pop)':>14}")
    for k in cfg.eval_ks:
        print(f"Recall@{k:<5}{val[k]['recall']:>10.4f}{test[k]['recall']:>10.4f}"
              f"{test[k]['pop_recall']:>14.4f}")
        print(f"NDCG@{k:<7}{val[k]['ndcg']:>10.4f}{test[k]['ndcg']:>10.4f}"
              f"{test[k]['pop_ndcg']:>14.4f}")


if __name__ == "__main__":
    main()
