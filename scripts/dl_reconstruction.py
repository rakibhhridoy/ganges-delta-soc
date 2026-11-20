"""
Deep-learning reconstruction (H3.2/H3.4).

IMPORTANT PROVENANCE NOTE. The original deep-learning training code used during the
study was not preserved. This script is an INDEPENDENT RE-IMPLEMENTATION built
strictly from the architecture description in the Methods (it is therefore a
faithful reconstruction, not the exact original code). Its purpose is to make the
manuscript's qualitative H3 claim --- that none of the deep-learning architectures
or their ensemble surpasses gradient boosting under leak-free cross-validation at
this sample size --- reproducible. Absolute values will differ from any single run
because deep models are high-variance at n=100; the ranking (GBR >= DL under
location-grouped CV) is the reproducible result.

Architectures (from Methods): multi-relational graph attention network (three edge
types: spatial <=50 km, same geomorphological zone, same location at adjacent
depths; 4 heads, 3 layers), mixture-of-experts (4 experts, top-2 gating), tabular
ViT (soil/spatial/embedding modality tokens + CLS, 3-layer MHSA), and a
weight-constrained ensemble of these with gradient boosting. Dual-task objective
(Huber + 0.3*CE), AdamW (1e-3, 1e-4), 200 epochs, early-stopping patience 30,
seed 42. Reported under random 5-fold OOF (optimistic), location-grouped 5-fold, and
20 random location partitions (authoritative). v4 change: ensemble weights are
cross-fitted (no test labels), and the verdict is computed rather than printed.

Run:  python3 dl_reconstruction.py     (requires torch + torch_geometric)
"""
import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.optimize import minimize

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)

feat = pd.read_csv(DATA / "features_preprocessed.csv")
feat = feat.drop(columns=[c for c in feat.columns if c.strip().upper() == "OM"])
cols = list(feat.columns)
y = pd.read_csv(DATA / "target.csv").iloc[:, 0].values.astype(np.float32)
soc = pd.read_csv(DATA / "GangesSOC.csv"); soc.columns = [c.strip() for c in soc.columns]
loc = pd.factorize(soc["Location"].values)[0]
lat, lon, depth = soc["Lat"].values.astype(float), soc["Long"].values.astype(float), soc["Depth"].values
ZF = {"Tidal Active": "Tidal.csv", "Active Delta": "Active.csv",
      "Mature Delta": "Mature.csv", "Moribund Delta": "Moribund.csv"}
zmap = {}
for z, f in ZF.items():
    zc = pd.read_csv(DATA / f); zc.columns = [c.strip() for c in zc.columns]
    for L in zc["Location"].unique():
        zmap[L] = z
zone = pd.factorize(np.array([zmap[L] for L in soc["Location"].values]))[0]
X = feat.values.astype(np.float32); N, Fdim = X.shape
ycls = np.digitize(y, [15, 50, 150]).astype(np.int64)
SOIL = ["Bulk Density", "Moisture", "Sand", "Silt", "Clay", "pH", "EC", "TN", "CEC"]
SPAT = ["Lat", "Long", "Elevation", "Slope", "Aspect"]
gi_soil = [cols.index(c) for c in SOIL]; gi_spat = [cols.index(c) for c in SPAT]
gi_emb = [i for i, c in enumerate(cols) if c.startswith("embedding")]
print(f"{N} samples, {Fdim} feats | groups soil={len(gi_soil)} spat={len(gi_spat)} emb={len(gi_emb)}")

def hav(i, j):
    R = 6371.; p1, p2 = np.radians(lat[i]), np.radians(lat[j])
    dp, dl = np.radians(lat[j] - lat[i]), np.radians(lon[j] - lon[i])
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
def edges(nodes):
    s, z, d = [], [], []; nl = list(nodes); ix = {n: k for k, n in enumerate(nl)}
    for a in range(len(nl)):
        for b in range(a + 1, len(nl)):
            i, j = nl[a], nl[b]
            if hav(i, j) <= 50: s += [[ix[i], ix[j]], [ix[j], ix[i]]]
            if zone[i] == zone[j]: z += [[ix[i], ix[j]], [ix[j], ix[i]]]
            if loc[i] == loc[j] and abs(int(depth[i]) - int(depth[j])) <= 20: d += [[ix[i], ix[j]], [ix[j], ix[i]]]
    te = lambda e: torch.tensor(e, dtype=torch.long).t().contiguous() if e else torch.empty((2, 0), dtype=torch.long)
    return te(s), te(z), te(d)

class GAT(nn.Module):
    def __init__(s, fin, h=16, heads=4, L=3):
        super().__init__(); s.L = nn.ModuleList(); d = fin
        for _ in range(L):
            s.L.append(nn.ModuleList([GATConv(d, h, heads=heads, concat=True) for _ in range(3)])); d = h * heads
        s.reg = nn.Linear(d, 1); s.cls = nn.Linear(d, 4)
    def forward(s, x, es, ez, ed):
        for r in s.L: x = F.elu(r[0](x, es) + r[1](x, ez) + r[2](x, ed))
        return s.reg(x).squeeze(-1), s.cls(x)

class MoE(nn.Module):
    def __init__(s, fin, k=4, hid=64):
        super().__init__(); s.k = k
        s.exp = nn.ModuleList([nn.Sequential(nn.Linear(fin, hid), nn.ReLU(), nn.Linear(hid, 32), nn.ReLU()) for _ in range(k)])
        s.gate = nn.Linear(fin, k); s.reg = nn.Linear(32, 1); s.cls = nn.Linear(32, 4)
    def forward(s, x):
        g = s.gate(x); top = torch.topk(g, 2, dim=-1); w = torch.zeros_like(g).scatter(-1, top.indices, F.softmax(top.values, -1))
        h = sum(w[:, i:i + 1] * s.exp[i](x) for i in range(s.k))
        return s.reg(h).squeeze(-1), s.cls(h)

class TabViT(nn.Module):
    def __init__(s, d=64, heads=4, L=3):
        super().__init__()
        s.ps = nn.Linear(len(gi_soil), d); s.pp = nn.Linear(len(gi_spat), d); s.pe = nn.Linear(len(gi_emb), d)
        s.cls_tok = nn.Parameter(torch.randn(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True); s.tr = nn.TransformerEncoder(enc, L)
        s.reg = nn.Linear(d, 1); s.cls = nn.Linear(d, 4)
    def forward(s, x):
        t = torch.stack([s.ps(x[:, gi_soil]), s.pp(x[:, gi_spat]), s.pe(x[:, gi_emb])], 1)
        b = x.shape[0]; t = torch.cat([s.cls_tok.expand(b, -1, -1), t], 1); h = s.tr(t)[:, 0]
        return s.reg(h).squeeze(-1), s.cls(h)

hub, ce = nn.HuberLoss(), nn.CrossEntropyLoss()
def fit_tab(ctor, Xtr, ytr, yctr):
    torch.manual_seed(SEED); m = ctor(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); yc = torch.tensor(yctr)
    best, bs, wait = 1e18, None, 0
    for _ in range(200):
        m.train(); opt.zero_grad(); pr, pc = m(Xt); l = hub(pr, yt) + 0.3 * ce(pc, yc); l.backward(); opt.step()
        if l.item() < best - 1e-4: best, bs, wait = l.item(), {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 30: break
    if bs: m.load_state_dict(bs)
    return m
def pred_tab(m, Xte):
    m.eval()
    with torch.no_grad(): pr, _ = m(torch.tensor(Xte)); return pr.numpy()

def gnn_preds(tr, te, inductive):
    sc = StandardScaler().fit(X[tr]); Xs = torch.tensor(sc.transform(X), dtype=torch.float32)
    yt = torch.tensor(y); yc = torch.tensor(ycls); torch.manual_seed(SEED)
    m = GAT(Fdim); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    if inductive:
        trn = np.array(sorted(tr)); es, ez, ed = edges(trn); Xtr = Xs[trn]; ytr, yctr = yt[trn], yc[trn]
        esF, ezF, edF = edges(np.arange(N))
    else:
        es, ez, ed = edges(np.arange(N)); trm = torch.zeros(N, dtype=torch.bool); trm[tr] = True
    best, bs, wait = 1e18, None, 0
    for _ in range(200):
        m.train(); opt.zero_grad()
        if inductive: pr, pc = m(Xtr, es, ez, ed); l = hub(pr, ytr) + 0.3 * ce(pc, yctr)
        else: pr, pc = m(Xs, es, ez, ed); l = hub(pr[trm], yt[trm]) + 0.3 * ce(pc[trm], yc[trm])
        l.backward(); opt.step()
        if l.item() < best - 1e-4: best, bs, wait = l.item(), {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 30: break
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad():
        if inductive: pr, _ = m(Xs, esF, ezF, edF); return pr[te].numpy()
        pr, _ = m(Xs, es, ez, ed); return pr[te].numpy()

def tab_oof(ctor, splits):
    oof = np.full(N, np.nan)
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        m = fit_tab(ctor, sc.transform(X[tr]).astype(np.float32), y[tr], ycls[tr])
        oof[te] = pred_tab(m, sc.transform(X[te]).astype(np.float32))
    return oof
def gbr_oof(splits):
    oof = np.full(N, np.nan)
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr]); g = GradientBoostingRegressor(random_state=SEED).fit(sc.transform(X[tr]), y[tr])
        oof[te] = g.predict(sc.transform(X[te]))
    return oof
def gnn_oof(splits, inductive):
    oof = np.full(N, np.nan)
    for tr, te in splits: oof[te] = gnn_preds(tr, te, inductive)
    return oof

def _weights(P, yy):
    f = lambda w: -r2_score(yy, P @ w)
    return minimize(f, [.25] * 4, method="SLSQP", bounds=[(0, 1)] * 4,
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1}).x

def ens(oofs, splits):
    """Weight-constrained ensemble with CROSS-FITTED weights: the weights applied to fold k are
    fitted on the out-of-fold predictions of the other folds only, so no test label is used.
    (v3 fitted the weights on all out-of-fold predictions, which used the test labels.)"""
    P = np.column_stack([oofs[k] for k in ["GBR", "GNN", "MoE", "ViT"]])
    out = np.full(N, np.nan)
    for tr, te in splits:
        out[te] = P[te] @ _weights(P[tr], y[tr])
    return r2_score(y, out), _weights(P, y)

def partition(seed, k=5):
    L = np.unique(loc); r = np.random.RandomState(seed); r.shuffle(L)
    return [(np.where(~np.isin(loc, f))[0], np.where(np.isin(loc, f))[0]) for f in np.array_split(L, k)]

def run(splits, inductive, tag, verbose=True):
    o = {"GBR": gbr_oof(splits), "GNN": gnn_oof(splits, inductive),
         "MoE": tab_oof(lambda: MoE(Fdim), splits), "ViT": tab_oof(lambda: TabViT(), splits)}
    r = {k: r2_score(y, o[k]) for k in o}
    r["Ensemble"], w = ens(o, splits)
    if verbose:
        print(f"\n=== {tag} ===")
        for k in r: print(f"  {k:8s} R^2 = {r[k]:+.3f}")
        print(f"  full-data weights (descriptive only) GBR/GNN/MoE/ViT = {np.round(w, 2)}")
        best = max((k for k in r if k != "GBR"), key=r.get)
        verdict = "is not surpassed" if r[best] <= r["GBR"] else f"IS SURPASSED by {best}"
        print(f"  -> on this split gradient boosting {verdict} (best other: {best} {r[best]:+.3f}).")
    return r

random5 = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(N)))
grouped = list(GroupKFold(5).split(np.arange(N), y, loc))
run(random5, False, "[A] random 5-fold OOF (optimistic, leaky)")
run(grouped, True, "[B] location-grouped 5-fold (GroupKFold)")

# [C] A single location split is high-variance at 20 locations; repeat over random location partitions.
REPS = 20
print(f"\n=== [C] {REPS} random location partitions (5-fold), inductive GNN ===")
keys = ["GBR", "GNN", "MoE", "ViT", "Ensemble"]
A = np.array([[run(partition(s), True, "", verbose=False)[k] for k in keys] for s in range(REPS)])
for i, k in enumerate(keys):
    line = f"  {k:8s} median {np.median(A[:, i]):+.3f}  range {A[:, i].min():+.3f} to {A[:, i].max():+.3f}"
    if k != "GBR": line += f"  beats GBR in {(A[:, i] > A[:, 0]).sum()}/{REPS}"
    print(line)
print("\nIndependent reconstruction from the Methods; high-variance at n=100.")
