"""
Adabiyotdagi usullarni shu datasetda sinash (notebooks/02 ning davomi).
Har bir tajriba bitta maqoladan olingan konkret retsept. Natijalar outputs/experiments_papers.csv ga yoziladi.

Ishga tushirish (loyiha ildizidan): python src/exp_papers.py
Talab: outputs/cache/*.parquet (02_experiments.ipynb yoki train.py hosil qiladi) — bo'lmasa o'zi hisoblaydi.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from train import features_windows, features_histograms, features_relative, LGB_PARAMS, BURST_CUTOFF_DAYS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from scipy.stats import rankdata

DATA, OUT, CACHE = "data", "outputs", "outputs/cache"
os.makedirs(CACHE, exist_ok=True)
tr = pd.read_csv(f"{DATA}/train_signals.csv", parse_dates=["signal_sanasi"]); y = tr.eskalatsiya.values
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
RES = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def recall_at_fpr(y, p, fpr_target=0.20):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(fpr_target, fpr, tpr))
def record(key, name, paper, oof, note):
    a, r = roc_auc_score(y, oof), recall_at_fpr(y, oof)
    RES.append(dict(key=key, name=name, paper=paper, auc=round(a, 4), recall20=round(r, 3), note=note))
    log(f"{name:60s} AUC={a:.4f}  recall@20%FPR={r:.3f}")
    pd.DataFrame(RES).to_csv(f"{OUT}/experiments_papers.csv", index=False)
    np.save(f"{OUT}/cache/oof_{key}.npy", oof)

def cached(name, fn):
    p = f"{CACHE}/{name}.parquet"
    if os.path.exists(p): return pd.read_parquet(p)
    F = fn(tr, f"{DATA}/train_transactions.parquet"); F.to_parquet(p); return F
clean = lambda X: X.loc[:, ~X.columns.duplicated()].replace([np.inf, -np.inf], np.nan).astype("float32")
X_all = clean(pd.concat([cached("windows", features_windows), cached("histograms", features_histograms), cached("relative", features_relative)], axis=1))
TOP = open(f"{OUT}/selected_features.txt").read().split()
X_top = X_all[TOP]
log(f"features: all {X_all.shape}, top {X_top.shape}")
imp = lambda X: SimpleImputer(strategy="median").fit_transform(X)

def cv(model_fn, X, proba=True, fit_kw=None):
    oof = np.zeros(len(y))
    for tri, vai in FOLDS:
        m = model_fn(); m.fit(X[tri] if isinstance(X, np.ndarray) else X.iloc[tri], y[tri], **(fit_kw or {}))
        Xv = X[vai] if isinstance(X, np.ndarray) else X.iloc[vai]
        oof[vai] = m.predict_proba(Xv)[:, 1] if proba else m.decision_function(Xv)
    return oof

# ---------------------------------------------------------------- 0. bizning yakuniy model (taqqoslash uchun)
oof_lgb = cv(lambda: lgb.LGBMClassifier(n_estimators=1500, random_state=42, **LGB_PARAMS), X_top)
oof_lr = cv(lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)), X_top)
record("ours", "Bizning model: LightGBM top-60 + LogReg rank-blend", "—", 0.7 * rankdata(oof_lgb) + 0.3 * rankdata(oof_lr), "yakuniy submission")

# ---------------------------------------------------------------- 1. Jullum et al. 2020 (DNB): XGBoost + "max, total, count per transaction type"
import xgboost as xgb
tx = pd.read_parquet(f"{DATA}/train_transactions.parquet").merge(tr[["signal_id", "signal_sanasi"]], on="signal_id")
tx["d"] = (tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds() / 86400; tx = tx[tx.d > BURST_CUTOFF_DAYS]
tx["grp"] = tx.tranzaksiya_turi + "_" + tx.kirim_chiqim
J = pd.concat([tx.pivot_table(index="signal_id", columns="grp", values="miqdor_indeksi", aggfunc=a).add_prefix(f"{a}_") for a in ["max", "sum", "count"]], axis=1)
J = J.reindex(tr.signal_id).fillna(0).astype("float32"); del tx
XGB = dict(n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.5, reg_lambda=5, eval_metric="auc", n_jobs=4, random_state=42)
record("jullum", "Jullum 2020 retsepti: XGBoost, tur bo'yicha max/total/count (24 feature)", "Jullum et al. 2020, JMLC",
       cv(lambda: xgb.XGBClassifier(**XGB), J), "DNB maqolasidagi feature oilasi: 'maximum and total amount, and the number of transactions of each transaction type'")
record("jullum_full", "Jullum 2020 modeli (XGBoost) bizning top-60 featurelarda", "Jullum et al. 2020",
       cv(lambda: xgb.XGBClassifier(**XGB), X_top), "algoritm farqi LightGBM vs XGBoost")

# ---------------------------------------------------------------- 2. Bakry et al. 2023 (ASXAML): XGBoost + RFECV + Optuna
t = time.time()
rfe = RFE(xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, n_jobs=4, random_state=42), n_features_to_select=40, step=0.2)
rfe.fit(imp(X_top), y); cols_rfe = list(X_top.columns[rfe.support_])
log(f"RFE -> {len(cols_rfe)} feature ({time.time()-t:.0f}s)")
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xr = X_top[cols_rfe]
    def objective(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 1200), learning_rate=trial.suggest_float("lr", 0.005, 0.05, log=True),
                 max_depth=trial.suggest_int("max_depth", 2, 5), subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample", 0.2, 0.8), reg_lambda=trial.suggest_float("lambda", 1, 30, log=True),
                 min_child_weight=trial.suggest_int("mcw", 5, 100), n_jobs=4, random_state=42)
        o = np.zeros(len(y))
        for tri, vai in FOLDS[:3]:
            m = xgb.XGBClassifier(**p).fit(Xr.iloc[tri], y[tri]); o[vai] = m.predict_proba(Xr.iloc[vai])[:, 1]
        idx = np.concatenate([v for _, v in FOLDS[:3]]); return roc_auc_score(y[idx], o[idx])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)); st.optimize(objective, n_trials=25)
    bp = st.best_params; p = dict(n_estimators=bp["n_estimators"], learning_rate=bp["lr"], max_depth=bp["max_depth"], subsample=bp["subsample"],
                                  colsample_bytree=bp["colsample"], reg_lambda=bp["lambda"], min_child_weight=bp["mcw"], n_jobs=4, random_state=42)
    record("asxaml", "Bakry 2023 (ASXAML): XGBoost + RFE (40 feature) + Optuna (25 trial)", "Bakry et al. 2023, J. Supercomputing",
           cv(lambda: xgb.XGBClassifier(**p), Xr), f"best params: {json.dumps(bp)}")
except ImportError:
    log("optuna yo'q")

# ---------------------------------------------------------------- 3. Feedzai (Eddin et al. 2021): LightGBM, ikki oyna nisbatlari, recall@20%FPR
# ularning feature retsepti: sum/mean/min/max/count per window + ratios & differences between two windows
fz = [c for c in X_all.columns if c.startswith(("w7_", "w30_", "w90_", "old_", "nb_")) and any(s in c for s in ["_n", "_amt_mean", "_amt_min", "_amt_max", "_amt_sum"])]
FZ = X_all[fz].copy()
for a, b in [("w7", "w30"), ("w30", "w90"), ("w30", "old"), ("w90", "old")]:
    for s in ["_n", "_amt_mean", "_amt_sum", "_amt_max"]:
        if f"{a}{s}" in FZ and f"{b}{s}" in FZ:
            FZ[f"ratio_{a}_{b}{s}"] = FZ[f"{a}{s}"] / (FZ[f"{b}{s}"].abs() + 1e-3); FZ[f"diff_{a}_{b}{s}"] = FZ[f"{a}{s}"] - FZ[f"{b}{s}"]
FZ = clean(FZ)
record("feedzai", f"Feedzai 2021 retsepti: LightGBM, oyna agregatlari + ikki oyna nisbat/farqlari ({FZ.shape[1]} feature)", "Eddin et al. 2021, arXiv:2112.07508",
       cv(lambda: lgb.LGBMClassifier(n_estimators=1500, random_state=42, **LGB_PARAMS), FZ), "graf featurelari yo'q (datasetda mijozlar orasida bog'lanish yo'q)")

# ---------------------------------------------------------------- 4. Anomaliya usullari (Oztas 2024 tavsiyasi; Paula 2016 autoencoder)
Xs = StandardScaler().fit_transform(imp(X_top))
iso = IsolationForest(n_estimators=500, contamination="auto", random_state=42).fit(Xs)
record("iforest", "Isolation Forest anomaliya skori (nazoratsiz, label ishlatilmagan)", "Oztas et al. 2024, FGCS (anomaly detection tavsiyasi)",
       -iso.score_samples(Xs), "label'siz skor; eskalatsiya = anomaliya degan gipoteza")
ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), max_iter=300, random_state=42, early_stopping=True).fit(Xs, Xs)
rec_err = ((ae.predict(Xs) - Xs) ** 2).mean(1)
record("autoenc", "Autoencoder (60→32→8→32→60) rekonstruksiya xatosi", "Paula et al. 2016, ICMLA",
       rec_err, "nazoratsiz; Braziliya eksport AML ishidagi g'oya")
pca = PCA(n_components=8, random_state=42).fit(Xs); rec_pca = ((pca.inverse_transform(pca.transform(Xs)) - Xs) ** 2).mean(1)
record("pca", "PCA (8 komponent) rekonstruksiya xatosi", "klassik anomaliya usuli", rec_pca, "nazoratsiz")
Xaug = np.column_stack([imp(X_top), -iso.score_samples(Xs), rec_err, rec_pca])
record("lgb_anom", "LightGBM top-60 + 3 ta anomaliya skori feature sifatida", "gibrid: nazoratli + nazoratsiz",
       cv(lambda: lgb.LGBMClassifier(n_estimators=1500, random_state=42, **LGB_PARAMS), Xaug), "anomaliya skorlari nazoratli modelga qo'shimcha beradimi?")

# ---------------------------------------------------------------- 5. Klassik AML klassifikatorlari (Chen et al. 2018 sharhi: SVM, RF, NB, NN)
Xi = imp(X_top)
record("nb", "Gaussian Naive Bayes", "Bakry 2023 da taqqoslangan bazaviy model", cv(lambda: GaussianNB(), Xi), "")
record("rf", "Random Forest (500 daraxt)", "Chen et al. 2018 sharhi", cv(lambda: RandomForestClassifier(500, min_samples_leaf=20, max_features=0.3, n_jobs=4, random_state=42), Xi), "")
record("et", "Extra Trees (500 daraxt)", "Chen et al. 2018 sharhi", cv(lambda: ExtraTreesClassifier(500, min_samples_leaf=20, max_features=0.3, n_jobs=4, random_state=42), Xi), "")
record("mlp", "MLP neyron tarmoq (128-64), standartlashtirilgan top-60", "Chen et al. 2018 sharhi (NN)",
       cv(lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-2, early_stopping=True, max_iter=300, random_state=42)), Xi), "")
from sklearn.svm import SVC
record("svm", "SVM (RBF), standartlashtirilgan top-60", "Chen et al. 2018 sharhi (SVM)",
       cv(lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=42)), Xi), "14k namunada ~10 daqiqa")

# ---------------------------------------------------------------- 6. Stacking (meta-learner)
base = {k: np.load(f"{OUT}/cache/oof_{k}.npy") for k in ["jullum_full", "feedzai", "rf", "mlp", "iforest", "autoenc"] if os.path.exists(f"{OUT}/cache/oof_{k}.npy")}
base["lgb"] = oof_lgb; base["lr"] = oof_lr
S = np.column_stack([rankdata(v) / len(y) for v in base.values()])
record("stack", f"Stacking: {len(base)} ta modelning OOF skorlari → logistik meta-model", "klassik ensemble",
       cv(lambda: LogisticRegression(C=1.0, max_iter=1000), S), "ikki darajali CV (OOF ustida)")

log("done"); print(pd.DataFrame(RES)[["name", "auc", "recall20"]].to_string())