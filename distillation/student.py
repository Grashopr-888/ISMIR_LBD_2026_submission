#!/usr/bin/env python
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/match_ab/student.py (changed: package imports, output directory, `restore` command, docstring)
"""Rung B: distill a student encoder, mel80 -> 64-d music2latent frame (offline, MPS).

Mandatory: MLP 80->256->256->64 (the Cortex-A7 budget estimate's shape, ~0.103 M params ~ 0.103 M MACs/frame,
i.e. ~1-2 % of the Stomp's 5-9 M MAC/frame allowance at 25 % of one A7 core).
Stretch:   GRU-128 (+dense 64) sequence variant  (`python -m distillation.student gru`).
Restore:   `python -m distillation.student restore` rebuilds the torch checkpoints that evaluate.py
           loads from the committed weights/*.json (the paper's students), exactly: re-exporting the
           restored models reproduces the JSON byte for byte.

Training data: (mel80 -> raw latent) pairs from the three packs' corpora + designated
training-only extras + MUSDB18-7 preview stems (data_prep.py train).  Val split is by FILE
(never pack files).  Loss = MSE + 0.3*(1 - cosine on mean-centred vectors) — the centred
direction is what mean-centred-cosine matching consumes.  Early stop on val loss.

Export: RTNeural-style JSON under $LBD_WORK/weights/ (training never overwrites the committed
weights/ of the repository) — dense layers as keras-layout
[kernel [n_in][n_out], bias [n_out]] with an explicit input standardization block (can be
folded into layer 0: W' = W/sigma, b' = b - W@(mu/sigma)).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from distillation import ab_common as C
from distillation import config as CFG

REPO_WEIGHTS = Path(__file__).resolve().parents[1] / "weights"   # the paper's students (CC BY-NC 4.0)
WEIGHTS = C.SCRATCH / "weights"                                 # where training exports
WEIGHTS.mkdir(parents=True, exist_ok=True)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(80, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 64))

    def forward(self, x):
        return self.net(x)


class GRUNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(80, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 64)

    def forward(self, x, h=None):                 # x [B,T,80]
        y, h = self.gru(x, h)
        return self.out(y), h


def composite_loss(pred, tgt, centre, w_cos=0.3):
    mse = torch.mean((pred - tgt) ** 2)
    a = pred - centre
    b = tgt - centre
    cos = torch.sum(a * b, -1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1) + 1e-8)
    return mse + w_cos * torch.mean(1.0 - cos), mse.detach(), torch.mean(1 - cos).detach()


def load_pairs():
    z = np.load(C.AB / "train" / "pairs.npz")
    X, Y, G, is_val = z["X"], z["Y"], z["G"], z["is_val"]
    tr = ~is_val
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    return X, Y, G, is_val, mu.astype(np.float32), sd.astype(np.float32)


def train_mlp():
    torch.manual_seed(CFG.SEED)
    X, Y, G, is_val, mu, sd = load_pairs()
    Xs = (X - mu) / sd
    dev = device()
    xt = torch.tensor(Xs[~is_val], device=dev)
    yt = torch.tensor(Y[~is_val], device=dev)
    xv = torch.tensor(Xs[is_val], device=dev)
    yv = torch.tensor(Y[is_val], device=dev)
    centre = yt.mean(0)
    model = MLP().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"MLP params: {n_par} (~{n_par/1e6:.3f} M MACs/frame); train {len(xt)} val {len(xv)} on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    best, best_state, patience, t0 = np.inf, None, 0, time.time()
    bs = 1024
    for ep in range(300):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            loss, _, _ = composite_loss(model(xt[idx]), yt[idx], centre)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vmse, vcos = composite_loss(model(xv), yv, centre)
        vl = float(vl)
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep % 10 == 0 or patience == 0 and ep < 5:
            print(f"  ep {ep:3d}  val {vl:.4f} (mse {float(vmse):.4f} cenc-cos-err {float(vcos):.4f})"
                  f"  best {best:.4f}  [{time.time()-t0:.0f}s]")
        if patience >= 25:
            print(f"  early stop at ep {ep} (best val {best:.4f})")
            break
    model.load_state_dict(best_state)
    model.cpu().eval()
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, C.AB / "student_mlp.pt")
    export_mlp_json(model, mu, sd, n_par)
    with torch.no_grad():
        pv = model(torch.tensor(Xs[is_val])).numpy()
    c = Y[~is_val].mean(0)
    a, b = pv - c, Y[is_val] - c
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    print(f"held-out(file) centred cos: mean {cos.mean():.4f}  median {np.median(cos):.4f}  "
          f"p10 {np.percentile(cos, 10):.4f}")


def export_mlp_json(model, mu, sd, n_par, path=None):
    L = [m for m in model.net if isinstance(m, nn.Linear)]
    layers = []
    for i, lin in enumerate(L):
        W = lin.weight.detach().numpy()            # torch [out, in]
        b = lin.bias.detach().numpy()
        layers.append({"type": "dense", "shape": [W.shape[1], W.shape[0]],
                       "activation": "relu" if i < len(L) - 1 else "linear",
                       "weights": [W.T.tolist(), b.tolist()]})   # keras/RTNeural [in][out]
    macs = sum(l["shape"][0] * l["shape"][1] for l in layers)
    doc = {
        "model": "t06_student_mlp_mel80_256_256_64",
        "framework": "torch->RTNeural-json (keras layout: kernel [n_in][n_out], bias [n_out])",
        "input": {
            "features": "log-mel 80 bands (ln(mel_power+1e-10)), 20 Hz-16 kHz, n_fft 4096, "
                        "start-aligned to the music2latent frame grid (~93 ms hop)",
            "standardize": {"mean": mu.tolist(), "std": sd.tolist(),
                            "note": "x_in = (logmel - mean)/std; foldable into layer 0: "
                                    "W' = W/std (per input), b' = b - (W/std)@mean"},
        },
        "output": "raw 64-d music2latent frame; match with pack centre/L2 as in match.py",
        "params": n_par, "macs_per_frame": macs,
        "budget_note": "kb/07: 5-9 M MACs/frame fp32 fits in 25% of one Cortex-A7 core "
                       f"@10.77 Hz; this net is {macs/1e6:.3f} M MACs (~{100*macs/7e6:.1f}% of "
                       "the mid-budget) via RTNeural fp32",
        "layers": layers,
    }
    p = Path(path) if path else WEIGHTS / "student_mlp_mel80.json"
    p.write_text(json.dumps(doc))
    print(f"exported {p} ({p.stat().st_size/1e3:.0f} kB, {n_par} params, {macs} MACs/frame)")


# ----------------------------------------------------------------------- GRU (stretch)
def train_gru():
    torch.manual_seed(CFG.SEED)
    X, Y, G, is_val, mu, sd = load_pairs()
    Xs = (X - mu) / sd
    dev = device()
    seqs = []
    for g in np.unique(G):
        m = G == g
        seqs.append((torch.tensor(Xs[m]), torch.tensor(Y[m]), bool(is_val[m][0])))
    tr = [s for s in seqs if not s[2]]
    va = [s for s in seqs if s[2]]
    centre = torch.cat([y for _, y, _ in tr]).mean(0).to(dev)
    model = GRUNet().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"GRU-128 params: {n_par}; {len(tr)} train / {len(va)} val files on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, patience = np.inf, None, 0
    rng = np.random.default_rng(CFG.SEED)
    for ep in range(200):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), 8):
            batch = [tr[j] for j in order[i:i + 8]]
            Tm = max(len(x) for x, _, _ in batch)
            xb = torch.zeros(len(batch), Tm, 80); yb = torch.zeros(len(batch), Tm, 64)
            mask = torch.zeros(len(batch), Tm, dtype=torch.bool)
            for k, (x, y, _) in enumerate(batch):
                xb[k, :len(x)] = x; yb[k, :len(y)] = y; mask[k, :len(x)] = True
            xb, yb, mask = xb.to(dev), yb.to(dev), mask.to(dev)
            pred, _ = model(xb)
            loss, _, _ = composite_loss(pred[mask], yb[mask], centre)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vls = []
            for x, y, _ in va:
                p, _ = model(x[None].to(dev))
                vl, _, _ = composite_loss(p[0], y.to(dev), centre)
                vls.append(float(vl) * len(x))
            vl = sum(vls) / sum(len(x) for x, _, _ in va)
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep % 5 == 0:
            print(f"  ep {ep:3d}  val {vl:.4f}  best {best:.4f}")
        if patience >= 15:
            print(f"  early stop at ep {ep}")
            break
    model.load_state_dict(best_state)
    model.cpu().eval()
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, C.AB / "student_gru.pt")
    export_gru_json(model, mu, sd, n_par)


def export_gru_json(model, mu, sd, n_par, path=None):
    """RTNeural keras-GRU layout (reset_after=True, gate order z,r,h; torch is r,z,n)."""
    H = model.gru.hidden_size
    wih = model.gru.weight_ih_l0.detach().numpy()      # [3H, in]  torch rows r|z|n
    whh = model.gru.weight_hh_l0.detach().numpy()
    bih = model.gru.bias_ih_l0.detach().numpy()
    bhh = model.gru.bias_hh_l0.detach().numpy()
    perm = np.r_[H:2 * H, 0:H, 2 * H:3 * H]            # r,z,n -> z,r,n(=h)
    kernel = wih[perm].T                               # [in, 3H]
    recurrent = whh[perm].T                            # [H, 3H]
    bias = np.stack([bih[perm], bhh[perm]])            # [2, 3H]  (input, recurrent)
    W, b = model.out.weight.detach().numpy(), model.out.bias.detach().numpy()
    macs = 3 * H * (80 + H) + 80 * 0 + H * 64
    doc = {"model": "t06_student_gru128_mel80",
           "framework": "torch->keras-GRU layout (reset_after=True, gates z,r,h)",
           "input": {"features": "log-mel 80 (same grid as MLP)",
                     "standardize": {"mean": mu.tolist(), "std": sd.tolist()}},
           "params": n_par, "macs_per_frame": macs,
           "layers": [{"type": "gru", "shape": [80, H],
                       "weights": [kernel.tolist(), recurrent.tolist(), bias.tolist()]},
                      {"type": "dense", "shape": [H, 64], "activation": "linear",
                       "weights": [W.T.tolist(), b.tolist()]}]}
    p = Path(path) if path else WEIGHTS / "student_gru128_mel80.json"
    p.write_text(json.dumps(doc))
    print(f"exported {p} ({p.stat().st_size/1e3:.0f} kB, {n_par} params, {macs} MACs/frame)")


# ----------------------------------------------------------- restore (added for the public repo)
def _f32(x):
    return np.asarray(x, dtype=np.float32)


def mlp_from_json(path=REPO_WEIGHTS / "student_mlp_mel80.json"):
    """Inverse of export_mlp_json: (model, mu, sd). float32 -> JSON -> float32 is lossless."""
    doc = json.loads(Path(path).read_text())
    m = MLP()
    idx = [i for i, mod in enumerate(m.net) if isinstance(mod, nn.Linear)]
    state = {}
    for i, L in zip(idx, doc["layers"]):
        state[f"net.{i}.weight"] = torch.tensor(_f32(L["weights"][0]).T.copy())   # [in][out] -> [out, in]
        state[f"net.{i}.bias"] = torch.tensor(_f32(L["weights"][1]))
    m.load_state_dict(state)
    st = doc["input"]["standardize"]
    return m.eval(), _f32(st["mean"]), _f32(st["std"])


def gru_from_json(path=REPO_WEIGHTS / "student_gru128_mel80.json"):
    """Inverse of export_gru_json (keras gate order z,r,h back to torch r,z,n)."""
    doc = json.loads(Path(path).read_text())
    g, d = doc["layers"]
    m = GRUNet()
    H = m.gru.hidden_size
    perm = np.r_[H:2 * H, 0:H, 2 * H:3 * H]           # its own inverse: swaps the first two gate blocks
    kernel, recurrent, bias = _f32(g["weights"][0]), _f32(g["weights"][1]), _f32(g["weights"][2])
    m.load_state_dict({"gru.weight_ih_l0": torch.tensor(kernel.T[perm].copy()),
                       "gru.weight_hh_l0": torch.tensor(recurrent.T[perm].copy()),
                       "gru.bias_ih_l0": torch.tensor(bias[0][perm].copy()),
                       "gru.bias_hh_l0": torch.tensor(bias[1][perm].copy()),
                       "out.weight": torch.tensor(_f32(d["weights"][0]).T.copy()),
                       "out.bias": torch.tensor(_f32(d["weights"][1]))})
    st = doc["input"]["standardize"]
    return m.eval(), _f32(st["mean"]), _f32(st["std"])


def restore():
    """Write $LBD_WORK/ab/student_{mlp,gru}.pt (what evaluate.py loads) from the committed JSON."""
    for name, build in (("student_mlp.pt", mlp_from_json), ("student_gru.pt", gru_from_json)):
        m, mu, sd = build()
        torch.save({"state": m.state_dict(), "mu": mu, "sd": sd}, C.AB / name)
        print(f"restored {C.AB / name} ({sum(p.numel() for p in m.parameters())} params)")


if __name__ == "__main__":
    if "restore" in sys.argv[1:]:
        restore()
    elif "gru" in sys.argv[1:]:
        train_gru()
    else:
        train_mlp()
