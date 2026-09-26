"""EDA sayti uchun barcha grafik ma'lumotlarini bitta site/data.js ga yig'adi (loyiha ildizidan: python src/site_data.py).

Har bir blok o'zining manba fayli bo'lmasa yoki hisoblashda xato chiqsa, mos maydonni
shunchaki qoldirib ketadi (yoki None yozadi) — index.html o'zi buni '—' deb ko'rsatadi,
sahifa hech qachon qulab tushmasligi kerak (qarang: README, "Har qadam ixtiyoriy")."""
import os, json, sys, re, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import roc_curve, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="#", help="site/index.html dagi GitHub havolasi uchun")
ARGS, _ = ap.parse_known_args()

def safe(fn, label):
    """Bitta blokni bajaradi; xato bo'lsa faylni to'xtatmay, ogohlantirib None qaytaradi."""
    try:
        return fn()
    except Exception as e:
        print(f"[site_data] OGOHLANTIRISH: '{label}' hisoblanmadi ({e!r}) — saytda '—' bo'ladi")
        return None

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
D["n_types"], D["n_dirs"] = len(D["type_share"]), len(D["dir_share"])
D["window_days"] = 90  # src/train.py: 1/7/30/90 kunlik oynalar, >90 — "old" bazaviy guruh
D["date_min"] = tr.signal_sanasi.min().strftime("%Y-%m")
D["date_max"] = tr.signal_sanasi.max().strftime("%Y-%m")
D["type_esc_rate"] = (tx.groupby("tranzaksiya_turi").eskalatsiya.mean() * 100).round(2).to_dict()
D["dir_esc_rate"] = (tx.groupby("kirim_chiqim").eskalatsiya.mean() * 100).round(2).to_dict()
nb = tx[tx.d > 3 / 1440]
bins = np.linspace(-3, 5, 41)
D["amount_density"] = dict(x=[round(float(v), 2) for v in (bins[:-1] + 0.1)],
    dismissed=np.histogram(nb.miqdor_indeksi[nb.eskalatsiya == 0], bins, density=True)[0].round(4).tolist(),
    escalated=np.histogram(nb.miqdor_indeksi[nb.eskalatsiya == 1], bins, density=True)[0].round(4).tolist())
D["type_amt_by_class"] = nb.groupby(["eskalatsiya", "tranzaksiya_turi"]).miqdor_indeksi.mean().round(3).unstack().to_dict()
dc = nb.groupby(np.floor(nb.d)).size() / len(tr); D["activity"] = dict(x=[int(i) for i in dc.index], y=dc.values.round(3).tolist())
D["activity"]["peak"] = round(float(max(D["activity"]["y"])), 2)
D["activity"]["last"] = round(float(D["activity"]["y"][-1]), 2)
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
D["amt_min"] = round(min(g["min"] for g in D["group_floors"]), 2)
D["amt_max"] = round(max(g["max"] for g in D["group_floors"]), 2)
D["heat_dev"] = safe(lambda: round(float(np.std(np.array(D["hour_dow_ratio"]))), 2), "heat_dev")
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

# "Chegara" topilmasi (08_floor.png): signalning minimal hajmi (all_amt_min) desil D1 va D2 —
# eng past qiymatlar kutilganidan kamroq xavfli chiqadi (nomonoton). all_amt_min deciles'dan olamiz.
am = D["deciles"].get("all_amt_min")
D["floor"] = safe(lambda: dict(value=am["edges"][0], rate_at=am["rate"][0], rate_above=am["rate"][1]), "floor") if am else None

# "n_hyp" gipoteza kartalari — har
# Moliyaviy monitoring alertlarining eskalatsiya qilinishini oson bashorat qilish. Ko'rsatkich: ROC-AUC. EDA sayti: site/index.htmlbiri allaqachon hisoblangan bitta desil-featureга bog'langan;
# AUC >= 0.55 bo'lsa "hot" (signal bor) deb belgilanadi.
def _hypotheses():
    H = []
    for i, (key, meta) in enumerate(D["deciles"].items(), start=1):
        H.append(dict(id=f"H{i}", title=meta["label"], method="10 teng guruh (desil), har birining haqiqiy eskalatsiya darajasi",
                       kind="hot" if meta["auc"] >= 0.55 else "", lohi=[meta["auc"], meta["auc"]], delta=None, raw=False))
    return H
D["hypotheses"] = safe(_hypotheses, "hypotheses") or []

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

# --- run_summary.json: model/pipeline sozlamalari (jamoa, CV skorlar, LightGBM parametrlari) ---
D["run"] = safe(lambda: json.load(open(f"{OUT}/run_summary.json")), "run") or {}
D["repo"] = ARGS.repo

# --- loglar, so'zma-so'z (index.html ularni <pre> ichida ko'rsatadi) ---
def _read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None
D["logs"] = dict(train=_read(f"{OUT}/train_log.txt"), papers=_read(f"{OUT}/exp_papers.log"), gru=_read(f"{OUT}/gru_training.log"))

# --- GRU: eng yaxshi/oxirgi epoch AUC va loss, logdan parslanadi ---
def _gru():
    log = D["logs"]["gru"]
    if not log:
        return None
    epochs = re.findall(r"Train Loss:\s*([\d.]+)\s*\|\s*Valid AUC:\s*([\d.]+)", log)
    best = re.search(r"Eng yaxshi AUC:\s*([\d.]+)", log)
    if not epochs or not best:
        return None
    return dict(best_auc=float(best.group(1)), last_auc=float(epochs[-1][1]),
                first_loss=float(epochs[0][0]), last_loss=float(epochs[-1][0]))
D["gru"] = safe(_gru, "gru")

# --- n_papers: adabiyotda iqtibos qilingan NOYOB maqolalar soni (umumiy/baza tavsiflarisiz) ---
def _n_papers():
    pep = pd.read_csv(f"{OUT}/experiments_papers.csv")
    excl = {"—", "klassik anomaliya usuli", "gibrid: nazoratli + nazoratsiz", "klassik ensemble",
            "Bakry 2023 da taqqoslangan bazaviy model"}
    def norm(s):
        s = re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()  # "Chen ... sharhi (SVM)" -> "Chen ... sharhi"
        m = re.match(r"([A-ZÅØ][\w]*\s+et al\.?\s+\d{4})", s)  # "Jullum et al. 2020, JMLC" -> "Jullum et al. 2020"
        return m.group(1) if m else s
    return len({norm(p) for p in pep["paper"] if p not in excl})
D["n_papers"] = safe(_n_papers, "n_papers")

# --- yagona reyting jadvali (09-bo'lim): outputs/experiments.csv (bizning yo'l) +
#     outputs/experiments_papers.csv dagi klassik/nazoratsiz/maqola/ketma-ket modellar ---
def _leaderboard():
    LBD = []
    exp = pd.read_csv(f"{OUT}/experiments.csv")
    for _, r in exp.iterrows():
        LBD.append(dict(name=r["Yondashuv"], auc=float(r["CV ROC-AUC"]), cat="seq" if r["key"] == "gru" else "ours",
                         note=(r["Xulosa"] if isinstance(r["Xulosa"], str) and r["Xulosa"] else None)))
    cat_map = {"jullum": "paper", "jullum_full": "paper", "asxaml": "paper", "feedzai": "paper", "lgb_anom": "paper",
               "iforest": "unsup", "autoenc": "unsup", "pca": "unsup",
               "nb": "classic", "rf": "classic", "et": "classic", "mlp": "classic", "svm": "classic", "stack": "classic"}
    pep = pd.read_csv(f"{OUT}/experiments_papers.csv")
    for _, r in pep.iterrows():
        if r["key"] in cat_map:
            LBD.append(dict(name=r["name"], auc=float(r["auc"]), cat=cat_map[r["key"]],
                             note=(r["note"] if isinstance(r["note"], str) and r["note"] else None)))
    return LBD
D["leaderboard"] = safe(_leaderboard, "leaderboard") or []

# --- reyting jadvaliga bog'liq xulosa raqamlari ---
def _lb_top6_spread():
    aucs = sorted((r["auc"] for r in D["leaderboard"]), reverse=True)[:6]
    return round(aucs[0] - aucs[-1], 4) if len(aucs) >= 2 else None
D["lb_top6_spread"] = safe(_lb_top6_spread, "lb_top6_spread")

def _model_spread():
    aucs = [r["auc"] for r in D["leaderboard"] if r["cat"] == "classic"]
    if D["run"].get("cv_lgb") is not None: aucs.append(D["run"]["cv_lgb"])
    if D["run"].get("cv_lr") is not None: aucs.append(D["run"]["cv_lr"])
    return round((max(aucs) - min(aucs)) / 2, 4) if len(aucs) >= 2 else None
D["model_spread"] = safe(_model_spread, "model_spread")

D["feature_gain"] = safe(lambda: round(D["run"]["cv_lgb"] - D["run"]["cv_all_features"], 4), "feature_gain")

os.makedirs("site", exist_ok=True)
open("site/data.js", "w").write("window.SITE_DATA = " + json.dumps(D, ensure_ascii=False) + ";\n")
print("site/data.js", os.path.getsize("site/data.js") // 1024, "KB")