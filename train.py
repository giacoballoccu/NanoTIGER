"""
train.py -- STAGE 4: train RecGPT on Semantic ID sequences.

Each user history is flattened into Semantic ID tokens and the transformer is
trained with the plain language-modeling objective: predict the next token.
Because items *are* their Semantic IDs, "predict the next token" is literally
"predict (the meaning of) the next item".

Leave-one-out protocol (the sequential-recommendation standard): we hold out
each user's most recent item for evaluation, so training sees the history with
the final item removed.

Run:  python train.py
Out:  out/recgpt.pt
"""

import torch

from config import cfg, DATA, OUT
from common import get_device, seed_everything, load_json, count_params
from tokenizer import build_tokenizer
from model import GPTConfig, RecGPT


def build_batches(sequences, tok, block_size, device):
    """Each training example = one user history with the validation (-2) and
    test (-1) items removed, flattened to tokens and right-padded. Holding out
    the two most recent items is the temporal split; training on seq[:-2] keeps
    them out of the model's sight. Returns (X, Y) for next-token prediction."""
    X, Y = [], []
    for seq in sequences:
        if len(seq) < 3:                      # need train + val + test
            continue
        toks = tok.encode_sequence(seq[:-2])  # exclude val (-2) and test (-1)
        toks = toks[: block_size + 1]
        x = toks[:-1]
        y = toks[1:]
        pad = block_size - len(x)
        x = x + [tok.PAD] * pad
        y = y + [tok.PAD] * pad
        X.append(x)
        Y.append(y)
    X = torch.tensor(X, dtype=torch.long, device=device)
    Y = torch.tensor(Y, dtype=torch.long, device=device)
    return X, Y


def train_recgpt(sequences, tok, device, epochs=None, verbose=True):
    """Train a RecGPT on the given user sequences and return (model, gptcfg).
    Reused by train.py (all users) and the ablation (per-domain subsets)."""
    epochs = cfg.gpt_epochs if epochs is None else epochs
    block_size = cfg.max_seq_len * tok.D + 1
    X, Y = build_batches(sequences, tok, block_size, device)
    gptcfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=block_size,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        dropout=cfg.dropout, pad_token=tok.PAD,
    )
    model = RecGPT(gptcfg).to(device)
    if verbose:
        print(f"[train] {X.shape[0]:,} seqs | block_size={block_size} | "
              f"RecGPT {count_params(model):,} params")
    opt = model.configure_optimizers(cfg.weight_decay, cfg.gpt_lr, device)

    n = X.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, cfg.gpt_batch_size):
            idx = perm[s:s + cfg.gpt_batch_size]
            _, loss = model(X[idx], Y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d} | loss {total / n:.4f}")
    return model, gptcfg


def main():
    seed_everything(cfg.seed)
    device = get_device()

    sequences = load_json(DATA / "sequences.json")["train"]
    tok = build_tokenizer(load_json(DATA / "semantic_ids.json")["ids"])
    print(f"[train] {len(sequences):,} users | Semantic ID length D={tok.D} | "
          f"vocab={tok.vocab_size}")

    model, gptcfg = train_recgpt(sequences, tok, device)

    from checkpoints import save_recgpt
    save_recgpt(model, gptcfg, OUT / "recgpt.pt")
    print("[train] saved out/recgpt.pt")
    print("Next: python eval.py")


if __name__ == "__main__":
    main()
