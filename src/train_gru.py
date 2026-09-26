"""
Ketma-ketlik yondashuvi: bi-GRU + masked attention.
=================================================
Jamoaning dastlabki g'oyasi — tranzaksiyalarni ketma-ketlik sifatida modellashtirish.
Skript o'z logini `outputs/gru_training.log` ga yozadi; shu fayldan raqamlarni
`src/site_data.py` (sayt uchun) va `notebooks/02_experiments.ipynb` (tajribalar jadvali) o'qiydi.

Ishga tushirish (loyiha ildizidan):
    pip install torch
    python src/train_gru.py                    # 100 epoch, 80/20 split
    python src/train_gru.py --epochs 30        # tezroq (CPU uchun)
    python src/train_gru.py --max-signals 2000 # sinov uchun kichik to'plam

Dastlabki versiyadan farqi (xatolar tuzatilgan):
  * attention mask — padding pozitsiyalari softmax'dan chiqarilgan;
  * tranzaksiya turi uchun Embedding (oldin kategoriya kodi son sifatida berilgan edi);
  * signaldan oldingi oxirgi 3 daqiqadagi "burst" artefakti chiqarib tashlangan;
  * normalizatsiya statistikasi faqat train qismidan olinadi.

Arxitektura ataylab dastlabkiday qoldirilgan (bi-GRU 64 × 2 + attention), chunki maqsad —
ketma-ketlik modeli shu datasetda qancha bera olishini halol o'lchash.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import BURST_CUTOFF_DAYS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="outputs")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--valid-size", type=float, default=0.20)
ap.add_argument("--max-signals", type=int, default=0, help="0 = hammasi; sinov uchun kichik son")
ap.add_argument("--cpu", action="store_true", help="GPU bo'lsa ham CPU'da ishlatish")
A = ap.parse_args()

os.makedirs(f"{A.out}/cache", exist_ok=True)
LOG = f"{A.out}/gru_training.log"
open(LOG, "w", encoding="utf-8").write(
    f"$ python src/train_gru.py --epochs {A.epochs} --hidden {A.hidden} --layers {A.layers}\n")


def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError:
    log("XATO: torch o'rnatilmagan. `pip install torch` va qaytadan ishga tushiring.")
    sys.exit(1)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

torch.manual_seed(A.seed); np.random.seed(A.seed)
dev = torch.device("cpu" if A.cpu or not torch.cuda.is_available() else "cuda")
dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"

# --------------------------------------------------------------------------- ma'lumot
t0 = time.time()
sig = pd.read_csv(f"{A.data}/train_signals.csv", parse_dates=["signal_sanasi"])
if A.max_signals:
    sig = sig.sample(A.max_signals, random_state=A.seed).reset_index(drop=True)
tx = pd.read_parquet(f"{A.data}/train_transactions.parquet")
n_all = len(tx)
tx = tx.merge(sig[["signal_id", "signal_sanasi"]], on="signal_id")
tx["d"] = (tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds() / 86400.0
tx = tx[tx.d > BURST_CUTOFF_DAYS].sort_values(["signal_id", "tranzaksiya_vaqti"])

TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
tx["typ"] = pd.Categorical(tx.tranzaksiya_turi, categories=TYPES).codes.astype("int64")
assert tx.typ.min() >= 0, "tranzaksiya_turi da kutilmagan qiymat bor"
tx["is_out"] = (tx.kirim_chiqim == "chiqim").astype("float32")
tx["hour"] = (tx.tranzaksiya_vaqti.dt.hour / 23.0).astype("float32")
tx["dn"] = (tx.d / max(tx.d.max(), 1e-9)).astype("float32")

y = sig.set_index("signal_id").eskalatsiya
ids = sig.signal_id.astype(str).to_numpy()          # pyarrow tipidagi ustunni oddiy numpy massivga
tr_id, va_id = train_test_split(ids, test_size=A.valid_size, random_state=A.seed, stratify=y.reindex(ids).values)
tr_set = set(tr_id)

# normalizatsiya statistikasi faqat train signallaridan
m = tx.signal_id.isin(tr_set)
mu, sd = float(tx.loc[m, "miqdor_indeksi"].mean()), float(tx.loc[m, "miqdor_indeksi"].std())
tx["amt"] = ((tx.miqdor_indeksi - mu) / (sd + 1e-9)).astype("float32")

FEATS = ["amt", "dn", "hour", "is_out"]
SEQ, TYP = {}, {}
for s, g in tx.groupby("signal_id", sort=False):
    SEQ[s] = g[FEATS].to_numpy(dtype="float32")
    TYP[s] = g["typ"].to_numpy(dtype="int64")
ids = np.array([s for s in ids if s in SEQ])                       # tranzaksiyasi bor signallar
tr_id = np.array([s for s in tr_id if s in SEQ]); va_id = np.array([s for s in va_id if s in SEQ])
lab = y.reindex(ids).astype("float32").to_dict()
lens = {s: len(SEQ[s]) for s in ids}
del tx

log(f"Qurilma: {dev.type} ({dev_name})")
log(f"Tranzaksiyalar: {n_all:,} → burst'siz {sum(lens.values()):,}".replace(",", " "))
log(f"Signallar: {len(ids)} · train {len(tr_id)} / valid {len(va_id)} · "
    f"ketma-ketlik uzunligi {min(lens.values())}…{max(lens.values())} (mediana {int(np.median(list(lens.values())))})")
log(f"Ma'lumot tayyor: {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- batch
class IdentityDS(Dataset):
    """BucketSampler signal_id larni beradi, DataLoader ularni shu yerdan o'tkazadi."""
    def __init__(self, keys): self.n = len(keys)
    def __len__(self): return self.n
    def __getitem__(self, key): return key


def collate(keys):
    L = torch.tensor([lens[s] for s in keys], dtype=torch.long)
    T = int(L.max())
    X = torch.zeros(len(keys), T, len(FEATS))
    K = torch.zeros(len(keys), T, dtype=torch.long)
    for i, s in enumerate(keys):
        n = lens[s]
        X[i, :n] = torch.from_numpy(SEQ[s])
        K[i, :n] = torch.from_numpy(TYP[s])
    t = torch.tensor([lab[s] for s in keys], dtype=torch.float32)
    return X, K, L, t


class BucketSampler(Sampler):
    """Uzunligi yaqin signallarni bitta batchga yig'adi — padding kamayadi, tezlik ~3x."""
    def __init__(self, keys, batch, shuffle=True):
        self.keys = np.array(sorted(keys, key=lambda s: lens[s])); self.b = batch; self.shuffle = shuffle
    def __iter__(self):
        batches = [self.keys[i:i + self.b].tolist() for i in range(0, len(self.keys), self.b)]
        if self.shuffle: np.random.shuffle(batches)
        return iter(batches)
    def __len__(self): return int(np.ceil(len(self.keys) / self.b))


tr_dl = DataLoader(IdentityDS(tr_id), batch_sampler=BucketSampler(tr_id, A.batch, True), collate_fn=collate)
va_sampler = BucketSampler(va_id, 256, shuffle=False)
va_dl = DataLoader(IdentityDS(va_id), batch_sampler=va_sampler, collate_fn=collate)
va_order = [s for b in va_sampler for s in b]          # valid tartibi qat'iy: AUC shu tartibda hisoblanadi


# --------------------------------------------------------------------------- model
class GRUAttn(nn.Module):
    def __init__(self, n_feat, n_types, hidden, layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(n_types, 4)
        self.gru = nn.GRU(n_feat + 4, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.att = nn.Linear(hidden * 2, 1)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x, k, L):
        z = torch.cat([x, self.emb(k)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(z, L.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.gru(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)          # (B, T, 2H)
        mask = torch.arange(o.size(1), device=o.device)[None, :] < L.to(o.device)[:, None]
        score = self.att(o).squeeze(-1).masked_fill(~mask, -1e9)              # padding attention'dan chiqadi
        w = torch.softmax(score, dim=1)
        ctx = (o * w.unsqueeze(-1)).sum(1)                                    # (B, 2H)
        return self.fc(ctx).squeeze(-1)


net = GRUAttn(len(FEATS), len(TYPES), A.hidden, A.layers, A.dropout).to(dev)
n_par = sum(p.numel() for p in net.parameters())
opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=1e-5)
p_rate = float(np.mean([lab[s] for s in tr_id]))
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - p_rate) / p_rate], device=dev))
log(f"Model: bi-GRU(hidden={A.hidden}, layers={A.layers}) + masked attention · {n_par:,} parametr".replace(",", " "))
log(f"O'qitish boshlandi ({A.epochs} epoch)\n")

best, best_ep = 0.0, 0
yv_ord = np.array([lab[s] for s in va_order], dtype="float32")

for ep in range(1, A.epochs + 1):
    te = time.time(); net.train(); tot = 0.0; nb = 0
    for X, K, L, t in tr_dl:
        opt.zero_grad()
        loss = lossf(net(X.to(dev), K.to(dev), L), t.to(dev))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        tot += float(loss.item()); nb += 1
    net.eval(); preds = []
    with torch.no_grad():
        for X, K, L, t in va_dl:
            preds.append(torch.sigmoid(net(X.to(dev), K.to(dev), L)).cpu().numpy())
    p = np.concatenate(preds)
    auc = roc_auc_score(yv_ord, p)
    log(f"Epoch {ep}/{A.epochs} yakunlandi | Train Loss: {tot / max(nb, 1):.4f} | Valid AUC: {auc:.4f} | {time.time() - te:.0f}s")
    if auc > best:
        best, best_ep = auc, ep
        torch.save(net.state_dict(), f"{A.out}/gru_best.pt")
        np.save(f"{A.out}/cache/gru_valid_pred.npy", p)
        log(f"   Yangi eng yaxshi natija, model saqlandi (AUC: {best:.4f})")

log(f"\nO'qitish tugadi! Eng yaxshi AUC: {best:.4f} ({best_ep}-epoch) · {time.time() - t0:.0f}s")
log(f"Log: {LOG} · eng yaxshi model: {A.out}/gru_best.pt")