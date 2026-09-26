"""EDA sayti uchun barcha grafik ma'lumotlarini bitta site/data.js ga yig'adi (loyiha ildizidan: python src/site_data.py)."""
import os, json, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_curve, roc_auc_score
DATA, OUT, CACHE = "data", "outputs", "outputs/cache"
tr = pd.read_csv(f"{DATA}/train_signals.csv", parse_dates=["signal_sanasi"]); y = tr.set_index("signal_id").eskalatsiya
D = {}
D["n_train"], D["n_test"], D["pos_rate"], D["n_pos"] = len(tr), 6000, round(float(y.mean()), 4), int(y.sum())

tx = pd.read_parquet(f"{DATA}/train_transactions.parquet").merge(tr, on="signal_id")
tx["d"] = (tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds() / 86400
D["n_tx_train"], D["n_tx_test"] = len(tx), 3027575
ps = tx.groupby("signal_id").size()
D["tx_per_signal"] = dict(mean=round(float(ps.mean()), 1), median=int(ps.median()), min=int(ps.min()), max=int(ps.max()))
h, e = np.histogram(ps, bins=40); D["tx_per_signal_hist"] = dict(x=[round(float(v), 0) for v in e[:-1]], y=h.tolist())
m = tr.groupby(tr.signal_sanasi.dt.to_period("M")).eskalatsiya.agg(["size", "mean"])
D["monthly"] = dict(labels=[str(i) for i in m.index], n=m["size"].tolist(), rate=(m["mean"] * 100).round(1).tolist())
D["type_share"] = (tx.tranzaksiya_turi.value_counts(normalize=True) * 100).round(2).to_dict()
D["dir_share"] = (tx.kirim_chiqim.value_counts(normalize=True) * 100).round(2).to_dict()
D["type_esc_rate"] = (tx.groupby("tranzaksiya_turi").eskalatsiya.mean() * 100).round(2).to_dict()
D["dir_esc_rate"] = (tx.groupby("kirim_chiqim").eskalatsiya.mean() * 100).round(2).to_dict()
nb = tx[tx.d > 3 / 1440]
bins = np.linspace(-3, 5, 41)
D["amount_density"] = dict(x=[round(float(v), 2) for v in (bins[:-1] + 0.1)],
    dismissed=np.histogram(nb.miqdor_indeksi[nb.eskalatsiya == 0], bins, density=True)[0].round(4).tolist(),
    escalated=np.histogram(nb.miqdor_indeksi[nb.eskalatsiya == 1], bins, density=True)[0].round(4).tolist())
D["type_amt_by_class"] = nb.groupby(["eskalatsiya", "tranzaksiya_turi"]).miqdor_indeksi.mean().round(3).unstack().to_dict()
dc = nb.groupby(np.floor(nb.d)).size() / len(tr); D["activity"] = dict(x=[int(i) for i in dc.index], y=dc.values.round(3).tolist())
sec = (tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds()
hb, eb = np.histogram(sec[(sec >= 0) & (sec < 600)] / 60, bins=40); D["burst_hist"] = dict(x=eb[:-1].round(2).tolist(), y=hb.tolist())
D["burst"] = dict(share=round(float((tx.d <= 3 / 1440).mean() * 100), 2), n_signals=int(tx[tx.d <= 3 / 1440].signal_id.nunique()),
                  mean_n=round(float(tx[tx.d <= 3 / 1440].groupby("signal_id").size().mean()), 1))
hm = nb.groupby([nb.tranzaksiya_vaqti.dt.dayofweek, nb.tranzaksiya_vaqti.dt.hour]).size().unstack()
D["hour_dow"] = (hm / hm.values.sum() * 100).round(3).values.tolist()
hm1 = nb[nb.eskalatsiya == 1].groupby([nb.tranzaksiya_vaqti.dt.dayofweek, nb.tranzaksiya_vaqti.dt.hour]).size().unstack()
hm0 = nb[nb.eskalatsiya == 0].groupby([nb.tranzaksiya_vaqti.dt.dayofweek, nb.tranzaksiya_vaqti.dt.hour]).size().unstack()
D["hour_dow_ratio"] = ((hm1 / hm1.values.sum()) / (hm0 / hm0.values.sum())).round(3).values.tolist()
grp = nb.groupby(["tranzaksiya_turi", "kirim_chiqim"]).miqdor_indeksi.agg(["min", "max"]).round(3)
D["group_floors"] = [dict(grp=f"{a} · {b}", min=float(r["min"]), max=float(r["max"])) for (a, b), r in grp.iterrows()]
# sinf bo'yicha signal darajasidagi o'rtachalar
s = nb.groupby("signal_id").agg(n=("d", "size"), amt=("miqdor_indeksi", "mean"), out=("kirim_chiqim", lambda x: (x == "chiqim").mean()),
                                naqd=("tranzaksiya_turi", lambda x: (x == "naqd").mean()), xal=("tranzaksiya_turi", lambda x: (x == "xalqaro").mean()))
s["y"] = y.reindex(s.index); D["class_means"] = s.groupby("y").mean().round(4).to_dict()
del tx, nb

# desil explorer
X = pd.concat([pd.read_parquet(f"{CACHE}/{n}.parquet") for n in ["windows", "histograms", "relative"]], axis=1)
X = X.loc[:, ~X.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
feats = {
    "type_bank_otkazmasi_minus_naqd_mean": "bank o'tkazmasi − naqd (o'rtacha hajm)",
    "grp_bank_otkazmasi_chiqim_minus_karta_chiqim_q75": "bank chiqim − karta chiqim (75-kvantil)",
    "all_amt_min": "signalning minimal hajmi",
    "nb_n": "tranzaksiyalar soni",
    "nb_out_frac": "chiqim ulushi",
    "nb_naqd_frac": "naqd ulushi",
    "nb_xalqaro_frac": "xalqaro ulushi",
    "nb_night_frac": "tungi (00–06) ulushi",
    "ratio_w7_all": "oxirgi 7 kun faolligi / umumiy",
    "nb_amt_max": "eng katta tranzaksiya",
}
D["deciles"] = {}
for c, lab in feats.items():
    v = X[c]; ok = v.notna()
    q = pd.qcut(v[ok].rank(method="first"), 10, labels=False)
    r = (y.reindex(v[ok].index).groupby(q.values).mean() * 100).round(1).tolist()
    edges = [round(float(v[ok].quantile(k / 10)), 3) for k in range(11)]
    auc = roc_auc_score(y.reindex(v[ok].index), v[ok]); auc = max(auc, 1 - auc)
    D["deciles"][c] = dict(label=lab, rate=r, edges=edges, auc=round(float(auc), 3))

imp = pd.read_csv(f"{OUT}/feature_importance.csv", index_col=0).iloc[:, 0]
D["importance"] = dict(names=imp.head(20).index.tolist(), share=(imp.head(20) / imp.sum() * 100).round(2).tolist(), n_total=int(len(imp)))
oof = pd.read_csv(f"{OUT}/oof_predictions.csv")
fpr, tpr, _ = roc_curve(oof.eskalatsiya, oof.oof_blend); idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
D["roc"] = dict(fpr=fpr[idx].round(4).tolist(), tpr=tpr[idx].round(4).tolist(), auc=round(float(roc_auc_score(oof.eskalatsiya, oof.oof_blend)), 4),
                recall20=round(float(np.interp(0.2, fpr, tpr)), 3))
cal = oof.groupby(pd.qcut(oof.oof_blend, 10, labels=False)).eskalatsiya.mean() * 100; D["score_deciles"] = cal.round(1).tolist()
D["cv_report"] = open(f"{OUT}/cv_report.txt").read()
if os.path.exists(f"{OUT}/experiments_papers.csv"):
    D["papers_exp"] = pd.read_csv(f"{OUT}/experiments_papers.csv").to_dict("records")
os.makedirs("site", exist_ok=True)
open("site/data.js", "w").write("window.SITE_DATA = " + json.dumps(D, ensure_ascii=False) + ";\n")
print("site/data.js", os.path.getsize("site/data.js") // 1024, "KB")