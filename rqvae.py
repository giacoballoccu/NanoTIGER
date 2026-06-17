"""
rqvae.py -- STAGE 3: learn Semantic IDs with a Residual-Quantized VAE.

This is the core idea of the whole repo. A Semantic ID is a short tuple of
discrete codes, e.g. (37, 198, 4), that *names an item by its meaning*. Similar
items share leading codes; the codes are produced by quantizing the item's
content embedding.

How RQ-VAE works (this is the TIGER recipe):

    embedding x  --encoder-->  z (latent)
    z is quantized in L residual stages. At stage l we snap the current
    residual r_l to its nearest vector in codebook C_l, record that index c_l,
    and subtract it off:

        r_0 = z
        c_l = argmin_k || r_l - C_l[k] ||      (the l-th Semantic ID digit)
        r_{l+1} = r_l - C_l[c_l]

    The quantized latent is the sum of the chosen codewords, z_q = sum_l C_l[c_l],
    and a decoder reconstructs x_hat = decoder(z_q).

    Semantic ID of the item = (c_0, c_1, ..., c_{L-1}).

Because each stage quantizes the *leftover* of the previous one, the first code
captures coarse meaning and later codes refine it -- a coarse-to-fine,
prefix-shared code space. That prefix sharing is what later lets a transformer
generalize across items.

Training loss (standard VQ-VAE with a straight-through estimator):

    L = ||x - x_hat||^2                          (reconstruction)
      + sum_l ||sg[r_l] - C_l[c_l]||^2           (codebook / dictionary loss)
      + beta * sum_l ||r_l - sg[C_l[c_l]]||^2     (commitment loss)

where sg[.] is stop-gradient. Gradients reach the encoder by copying them
straight through the (non-differentiable) argmax.

Run:  python rqvae.py
Out:  data/semantic_ids.json   {"ids": [[c0,c1,c2], ...], "n_levels", "codebook_size"}
      out/rqvae.pt             the trained quantizer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg, DATA, OUT
from common import get_device, seed_everything, save_json, count_params


# ----------------------------------------------------------------------------
# A single vector-quantization codebook with straight-through gradients.
# ----------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, dim)
        # small uniform init; replaced by k-means on the first batch (see below)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self._initialized = False

    @torch.no_grad()
    def kmeans_init(self, x: torch.Tensor):
        """Seed the codebook from the data with a few Lloyd iterations. This is
        the standard trick that prevents dead codes early in training."""
        k = self.embedding.weight.shape[0]
        idx = torch.randperm(x.shape[0])[:k]
        centroids = x[idx].clone()
        if centroids.shape[0] < k:  # pad if fewer points than codes
            pad = centroids[torch.randint(0, centroids.shape[0], (k - centroids.shape[0],))]
            centroids = torch.cat([centroids, pad], 0)
        for _ in range(10):
            d = torch.cdist(x, centroids)
            assign = d.argmin(1)
            for c in range(k):
                pts = x[assign == c]
                if len(pts) > 0:
                    centroids[c] = pts.mean(0)
        self.embedding.weight.data.copy_(centroids)
        self._initialized = True

    @torch.no_grad()
    def revive_dead_codes(self, residual: torch.Tensor) -> int:
        """Random restart: any codeword no item currently maps to is moved on top
        of a real residual. This is the standard cure for codebook collapse --
        without it most codes die and the Semantic IDs lose resolution."""
        used = torch.unique(torch.cdist(residual, self.embedding.weight).argmin(1))
        k = self.embedding.weight.shape[0]
        dead = torch.ones(k, dtype=torch.bool, device=residual.device)
        dead[used] = False
        n_dead = int(dead.sum())
        if n_dead and residual.shape[0]:
            pick = torch.randint(0, residual.shape[0], (n_dead,), device=residual.device)
            self.embedding.weight.data[dead] = residual[pick]
        return n_dead

    def forward(self, r: torch.Tensor):
        # nearest codeword by Euclidean distance
        dist = torch.cdist(r, self.embedding.weight)      # (B, K)
        idx = dist.argmin(1)                              # (B,)
        q = self.embedding(idx)                           # (B, dim)

        codebook_loss = F.mse_loss(q, r.detach())         # move codes toward inputs
        commit_loss = F.mse_loss(r, q.detach())           # move encoder toward codes

        # straight-through: forward uses q, backward flows as if q == r
        q_st = r + (q - r).detach()
        return q_st, idx, codebook_loss, commit_loss


# ----------------------------------------------------------------------------
# Encoder / decoder MLPs.
# ----------------------------------------------------------------------------
def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        enc_sizes = [cfg.embed_dim, *cfg.rq_hidden, cfg.rq_latent_dim]
        dec_sizes = [cfg.rq_latent_dim, *reversed(cfg.rq_hidden), cfg.embed_dim]
        self.encoder = mlp(enc_sizes)
        self.decoder = mlp(dec_sizes)
        self.quantizers = nn.ModuleList(
            VectorQuantizer(cfg.rq_codebook_size, cfg.rq_latent_dim)
            for _ in range(cfg.rq_levels)
        )
        self.beta = cfg.rq_beta

    def quantize(self, z):
        """Residual quantization. Returns (z_q, list_of_codes, vq_loss)."""
        residual = z
        z_q = torch.zeros_like(z)
        codes, vq_loss = [], 0.0
        for vq in self.quantizers:
            q, idx, cb_loss, commit = vq(residual)
            residual = residual - q          # pass the leftover to the next level
            z_q = z_q + q
            codes.append(idx)
            vq_loss = vq_loss + cb_loss + self.beta * commit
        return z_q, torch.stack(codes, 1), vq_loss   # codes: (B, L)

    def forward(self, x):
        z = self.encoder(x)
        z_q, codes, vq_loss = self.quantize(z)
        x_hat = self.decoder(z_q)
        recon = F.mse_loss(x_hat, x)
        return x_hat, codes, recon, vq_loss

    @torch.no_grad()
    def encode_ids(self, x):
        """Item embeddings -> Semantic ID codes, (N, L) int tensor."""
        z = self.encoder(x)
        _, codes, _ = self.quantize(z)
        return codes


# ----------------------------------------------------------------------------
# Training.
# ----------------------------------------------------------------------------
def disambiguate(codes: np.ndarray) -> np.ndarray:
    """Two items can collide on the same (c0..c_{L-1}) tuple. Following TIGER,
    append an extra digit that counts collisions so every item gets a unique
    Semantic ID. Items with unique codes get a trailing 0."""
    seen = {}
    extra = np.zeros((codes.shape[0], 1), dtype=np.int64)
    for i, row in enumerate(map(tuple, codes)):
        extra[i, 0] = seen.get(row, 0)
        seen[row] = seen.get(row, 0) + 1
    return np.concatenate([codes, extra], axis=1)


@torch.no_grad()
def recon_loss(model, x):
    """Mean reconstruction loss on a set of items (no grad)."""
    was_training = model.training
    model.eval()
    _, _, recon, _ = model(x)
    if was_training:
        model.train()
    return recon.item()


def main():
    seed_everything(cfg.seed)
    device = get_device()

    emb = np.load(DATA / "item_emb.npy")
    x_all = torch.from_numpy(emb).float().to(device)
    n_all = x_all.shape[0]

    # Item-level train/val split. RQ-VAE "overfitting" = it reconstructs the
    # items it trained on but not held-out ones, so we watch the gap between
    # train and val reconstruction.
    perm = torch.randperm(n_all, device=device)
    n_val = max(1, int(cfg.rq_val_frac * n_all))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr, x_val = x_all[tr_idx], x_all[val_idx]
    print(f"[rqvae] {n_all:,} items (train {len(tr_idx):,} / val {len(val_idx):,}), "
          f"dim {x_all.shape[1]}")

    model = RQVAE(cfg).to(device)
    print(f"[rqvae] {count_params(model):,} params | "
          f"{cfg.rq_levels} levels x {cfg.rq_codebook_size} codes")

    # k-means init each codebook on the (train) residuals it will actually see
    with torch.no_grad():
        residual = model.encoder(x_tr)
        for vq in model.quantizers:
            vq.kmeans_init(residual)
            q, _, _, _ = vq(residual)
            residual = residual - q

    opt = torch.optim.Adam(model.parameters(), lr=cfg.rq_lr,
                           weight_decay=cfg.rq_weight_decay)
    n = x_tr.shape[0]
    for epoch in range(cfg.rq_epochs):
        # periodically restart dead codes (only while the codebook still settles)
        if epoch and epoch % cfg.rq_revive_every == 0 and epoch < 0.8 * cfg.rq_epochs:
            with torch.no_grad():
                residual = model.encoder(x_tr)
                for vq in model.quantizers:
                    vq.revive_dead_codes(residual)
                    q, _, _, _ = vq(residual)
                    residual = residual - q
        idx = torch.randperm(n, device=device)
        total_recon = total_vq = 0.0
        for s in range(0, n, cfg.rq_batch_size):
            batch = x_tr[idx[s:s + cfg.rq_batch_size]]
            _, _, recon, vq_loss = model(batch)
            (recon + vq_loss).backward()
            opt.step()
            opt.zero_grad()
            total_recon += recon.item() * len(batch)
            total_vq += float(vq_loss) * len(batch)
        if epoch % 20 == 0 or epoch == cfg.rq_epochs - 1:
            print(f"  epoch {epoch:3d} | train recon {total_recon / n:.4f} | "
                  f"val recon {recon_loss(model, x_val):.4f} | vq {total_vq / n:.4f}")

    # produce Semantic IDs for every item
    codes = model.encode_ids(x_all).cpu().numpy()

    # --- health checks -----------------------------------------------------
    tr_r, val_r = recon_loss(model, x_tr), recon_loss(model, x_val)
    print(f"[rqvae] reconstruction: train {tr_r:.4f} | val {val_r:.4f} | "
          f"gap {val_r - tr_r:+.4f}  (small gap = not overfitting)")
    for l in range(cfg.rq_levels):
        used = len(np.unique(codes[:, l]))
        print(f"[rqvae] codebook {l}: {used:>3}/{cfg.rq_codebook_size} codes used "
              f"({100 * used / cfg.rq_codebook_size:.0f}%)")
    n_unique = len({tuple(r) for r in codes})
    print(f"[rqvae] {n_unique:,}/{len(codes):,} unique code tuples "
          f"({100 * n_unique / len(codes):.1f}% before disambiguation)")
    codes = disambiguate(codes)  # adds a trailing uniqueness digit

    save_json(
        {
            "ids": codes.tolist(),
            "n_levels": cfg.rq_levels,
            "codebook_size": cfg.rq_codebook_size,
        },
        DATA / "semantic_ids.json",
    )
    from checkpoints import save_rqvae
    save_rqvae(model, cfg, OUT / "rqvae.pt")
    print(f"[rqvae] saved data/semantic_ids.json and out/rqvae.pt")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
