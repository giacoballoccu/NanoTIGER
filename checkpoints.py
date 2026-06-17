"""
checkpoints.py -- save / reload the trained models in one line.

The notebooks want to *load* the properly trained models (not retrain a toy in
the cell), so each checkpoint is self-describing: it stores the few config
numbers needed to rebuild the module, then the weights. Reloading needs only a
path and a device.

    save_rqvae(model, cfg, path);   model = load_rqvae(path, device)
    save_recgpt(model, gptcfg, path); model, gptcfg = load_recgpt(path, device)
"""

from types import SimpleNamespace

import torch

from rqvae import RQVAE
from model import GPTConfig, RecGPT

# the RQVAE constructor reads exactly these fields off its cfg object
_RQVAE_FIELDS = (
    "embed_dim", "rq_levels", "rq_codebook_size",
    "rq_latent_dim", "rq_hidden", "rq_beta",
)


def save_rqvae(model: RQVAE, cfg, path) -> None:
    torch.save(
        {"model": model.state_dict(),
         "cfg": {k: getattr(cfg, k) for k in _RQVAE_FIELDS}},
        path,
    )


def load_rqvae(path, device) -> RQVAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ckpt["cfg"])
    model = RQVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def save_recgpt(model: RecGPT, gptcfg: GPTConfig, path) -> None:
    torch.save({"model": model.state_dict(), "gptcfg": gptcfg.__dict__}, path)


def load_recgpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gptcfg = GPTConfig(**ckpt["gptcfg"])
    model = RecGPT(gptcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, gptcfg
