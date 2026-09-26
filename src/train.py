"""
Fintech track — alert escalation probability
============================================
Pipeline:  transactions ──> per-signal features ──> feature selection ──> LightGBM + LogReg blend ──> CSV

Usage (run from the project root):
    python src/train.py --team 1234                 # full run: CV + submission -> outputs/team_1234.csv
    python src/train.py --team 1234 --quick         # skip the extra CV pass
    python src/train.py --team 1234 --data data     # data folder (default: data)

Requirements:
    pip install pandas pyarrow numpy scikit-learn lightgbm scipy

Expected output (5-fold CV): LightGBM ≈ 0.66, LogReg ≈ 0.65, blend ≈ 0.66 ROC-AUC.
Runtime ≈ 10 min, peak RAM ≈ 5 GB.

Outputs (in outputs/):
    team_<ID>.csv              submission
    train_log.txt              this run's log
    run_summary.json           params, feature counts, CV scores (read by src/site_data.py)
    selected_features.txt      the features the final model uses
    feature_importance.csv     LightGBM gain for every candidate feature
    oof_predictions.csv        out-of-fold blend predictions for the training signals
    cv_report.txt              CV scores of this run
    train_log.txt              the full console log of this run
"""
import argparse, gc, itertools, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
DATA_DIR = "data"                     # overridden by --data
TEAM_ID = "XXXX"                      # overridden by --team
OUT_DIR = "outputs"
SEED = 42
N_FOLDS = 5
N_TOP_FEATURES = 60                   # keep the strongest features, drop the noise
N_SEEDS_FINAL = 10                    # seed-bagging for the final LightGBM
BLEND_W_LGB = 0.7                     # rank-blend weight for LightGBM (rest -> LogReg)
BURST_CUTOFF_DAYS = 3 / 1440          # last 3 minutes before the signal = data artefact
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
DIRS = ["kirim", "chiqim"]

LGB_PARAMS = dict(learning_rate=0.005, num_leaves=7, min_child_samples=100,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.3,
                  reg_lambda=10, reg_alpha=1, verbose=-1)
LGB_N_ESTIMATORS = 1500


LOG_FILE = None
SUMMARY = {}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------- helpers
def load_tx(sig: pd.DataFrame, path: str) -> pd.DataFrame:
    """Read a transaction parquet, attach the signal date, compute days-before-signal."""
    tx = pd.read_parquet(path).merge(sig[["signal_id", "signal_sanasi"]], on="signal_id")
    tx["d"] = ((tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds() / 86400.0).astype("float32")
    tx["amt"] = tx.miqdor_indeksi.astype("float32")
    tx["tranzaksiya_turi"] = tx.tranzaksiya_turi.astype("category")
    tx["kirim_chiqim"] = tx.kirim_chiqim.astype("category")
    return tx.drop(columns=["signal_sanasi", "miqdor_indeksi"])


# ----------------------------------------------------------------------------- feature family 1
def features_windows(sig: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Count / amount / direction / type statistics over several look-back windows:
    full history, last 1-7-30-90 days, the >90-day baseline, and the 3-minute "burst"
    right before the signal (kept separate — it is post-signal activity clipped to 23:57–00:00).
    """
    tx = load_tx(sig, path)
    tx["is_out"] = (tx.kirim_chiqim == "chiqim").astype("int8")
    tx["hour"] = tx.tranzaksiya_vaqti.dt.hour.astype("int8")
    tx["dow"] = tx.tranzaksiya_vaqti.dt.dayofweek.astype("int8")
    tx["night"] = (tx.hour < 6).astype("int8")
    tx["weekend"] = (tx.dow >= 5).astype("int8")
    tx["pos"] = (tx.amt > 0).astype("int8")
    tx["big"] = (tx.amt > 1.5).astype("int8")
    tx["huge"] = (tx.amt > 2.5).astype("int8")
    tx["neg_big"] = (tx.amt < -1.5).astype("int8")
    tx["sgn_amt"] = np.where(tx.is_out == 1, -tx.amt, tx.amt).astype("float32")
    tx["burst"] = (tx.d < BURST_CUTOFF_DAYS).astype("int8")

    def agg(df: pd.DataFrame, name: str) -> pd.DataFrame:
        g = df.groupby("signal_id", observed=True)
        a = {
            f"{name}_n": g.size(),
            f"{name}_amt_mean": g.amt.mean(), f"{name}_amt_std": g.amt.std(),
            f"{name}_amt_max": g.amt.max(), f"{name}_amt_min": g.amt.min(),
            f"{name}_amt_sum": g.amt.sum(), f"{name}_amt_med": g.amt.median(),
            f"{name}_amt_q90": g.amt.quantile(0.9), f"{name}_amt_q10": g.amt.quantile(0.1),
            f"{name}_amt_skew": g.amt.skew(),
            f"{name}_out_frac": g.is_out.mean(), f"{name}_out_n": g.is_out.sum(),
            f"{name}_sgn_sum": g.sgn_amt.sum(), f"{name}_pos_frac": g.pos.mean(),
            f"{name}_big_n": g.big.sum(), f"{name}_huge_n": g.huge.sum(), f"{name}_negbig_n": g.neg_big.sum(),
            f"{name}_night_frac": g.night.mean(), f"{name}_weekend_frac": g.weekend.mean(),
            f"{name}_d_mean": g.d.mean(), f"{name}_d_min": g.d.min(), f"{name}_d_max": g.d.max(), f"{name}_d_std": g.d.std(),
            f"{name}_hour_mean": g.hour.mean(), f"{name}_hour_std": g.hour.std(),
            f"{name}_nuniq_days": g.d.apply(lambda s: np.floor(s).nunique()),
        }
        for flag, lab in [(1, "out"), (0, "in")]:
            gd = df[df.is_out == flag].groupby("signal_id", observed=True).amt
            a[f"{name}_amt_{lab}_mean"] = gd.mean()
            a[f"{name}_amt_{lab}_max"] = gd.max()
        for t in TYPES:
            gt = df[df.tranzaksiya_turi == t].groupby("signal_id", observed=True).amt
            a[f"{name}_{t}_n"] = gt.size()
            a[f"{name}_{t}_amt_mean"] = gt.mean()
            a[f"{name}_{t}_amt_max"] = gt.max()
        A = pd.DataFrame(a)
        for t in TYPES:
            A[f"{name}_{t}_n"] = A[f"{name}_{t}_n"].fillna(0)
            A[f"{name}_{t}_frac"] = A[f"{name}_{t}_n"] / A[f"{name}_n"]
        return A

    parts = [agg(tx, "all")]
    nb = tx[tx.burst == 0]
    parts.append(agg(nb, "nb"))
    parts.append(agg(tx[tx.burst == 1], "burst"))
    for w in (1, 7, 30, 90):
        parts.append(agg(nb[nb.d < w], f"w{w}"))
    parts.append(agg(nb[nb.d >= 90], "old"))

    # inter-transaction gaps and daily-count regularity (non-burst only)
    s = nb.sort_values(["signal_id", "tranzaksiya_vaqti"])[["signal_id", "tranzaksiya_vaqti", "d"]].copy()
    s["gap"] = (s.groupby("signal_id", observed=True).tranzaksiya_vaqti.diff().dt.total_seconds() / 3600).astype("float32")
    g = s.groupby("signal_id", observed=True).gap
    parts.append(pd.DataFrame({"gap_mean": g.mean(), "gap_std": g.std(), "gap_max": g.max(),
                               "gap_med": g.median(), "gap_min": g.min()}))
    s["day"] = np.floor(s.d)
    g = s.groupby(["signal_id", "day"], observed=True).size().rename("c").reset_index().groupby("signal_id").c
    parts.append(pd.DataFrame({"daily_mean": g.mean(), "daily_std": g.std(), "daily_max": g.max(),
                               "daily_cv": g.std() / g.mean()}))
    del tx, nb, s; gc.collect()

    F = pd.concat(parts, axis=1).reindex(sig.signal_id)
    F["ratio_w7_all"] = F.w7_n / F.nb_n
    F["ratio_w30_all"] = F.w30_n / F.nb_n
    F["ratio_w1_w7"] = F.w1_n / F.w7_n
    F["amt_trend"] = F.w30_amt_mean - F.old_amt_mean
    F["out_trend"] = F.w30_out_frac - F.old_out_frac
    F["burst_ratio"] = F.burst_n / F.nb_n
    F["sig_month"] = sig.signal_sanasi.dt.month.values
    F["sig_dow"] = sig.signal_sanasi.dt.dayofweek.values
    F["sig_doy"] = sig.signal_sanasi.dt.dayofyear.values
    F["sig_year"] = sig.signal_sanasi.dt.year.values
    return F


# ----------------------------------------------------------------------------- feature family 2
def features_histograms(sig: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Shape of the amount distribution per signal: shares in fixed miqdor_indeksi bins and
    quantiles — overall, per type, per direction, per type×direction, and for the last 30 days.
    Plus linear trends of amount and outgoing-share over time.
    """
    tx = load_tx(sig, path)
    tx = tx[tx.d >= BURST_CUTOFF_DAYS].copy()
    bins = [-9, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 9]
    tx["b"] = pd.cut(tx.amt, bins, labels=False).astype("int8")
    out = {}

    def add(df, name):
        h = pd.crosstab(df.signal_id, df.b, normalize="index")
        for c in h.columns:
            out[f"{name}_hb{int(c)}"] = h[c]
        q = df.groupby("signal_id", observed=True).amt.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        for c in q.columns:
            out[f"{name}_q{int(c * 100)}"] = q[c]

    add(tx, "h_all")
    for t in TYPES:
        add(tx[tx.tranzaksiya_turi == t], f"h_{t}")
    for dr in DIRS:
        add(tx[tx.kirim_chiqim == dr], f"h_{dr}")
    for t in TYPES[:3]:
        for dr in DIRS:
            g = tx[(tx.tranzaksiya_turi == t) & (tx.kirim_chiqim == dr)].groupby("signal_id", observed=True).amt
            out[f"{t}_{dr}_n"] = g.size(); out[f"{t}_{dr}_mean"] = g.mean()
            out[f"{t}_{dr}_max"] = g.max(); out[f"{t}_{dr}_std"] = g.std()
    add(tx[tx.d < 30], "h_w30")

    tx["dd"] = (tx.d / 180).astype("float32")
    tx["is_out"] = (tx.kirim_chiqim == "chiqim").astype("float32")
    g = tx.groupby("signal_id", observed=True)
    var_dd = g.dd.var()
    out["amt_slope"] = g.apply(lambda x: np.cov(x.dd, x.amt)[0, 1] if len(x) > 5 else np.nan) / var_dd
    out["out_slope"] = g.apply(lambda x: np.cov(x.dd, x.is_out)[0, 1] if len(x) > 5 else np.nan) / var_dd
    del tx; gc.collect()
    return pd.DataFrame(out).reindex(sig.signal_id)


# ----------------------------------------------------------------------------- feature family 3
def features_relative(sig: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    THE strongest family. Amount statistics per type / direction / type×direction and,
    crucially, their pairwise differences (e.g. bank_otkazmasi − naqd mean).
    Alerts whose bank transfers are small relative to their own cash & card activity are
    escalated ~3x more often (28.7% in the lowest decile vs 10.2% in the highest).
    """
    tx = load_tx(sig, path)
    tx = tx[tx.d >= BURST_CUTOFF_DAYS].copy()
    tx["grp"] = (tx.tranzaksiya_turi.astype(str) + "_" + tx.kirim_chiqim.astype(str)).astype("category")
    out, stats = {}, {}
    STATS = ["mean", "median", "std", "q10", "q25", "q75", "q90"]

    for key, col in [("type", "tranzaksiya_turi"), ("dir", "kirim_chiqim"), ("grp", "grp")]:
        for stat in STATS:
            fn = stat if not stat.startswith("q") else (lambda s, q=int(stat[1:]) / 100: s.quantile(q))
            p = tx.pivot_table(index="signal_id", columns=col, values="amt", aggfunc=fn, observed=True)
            for c in p.columns:
                out[f"{key}_{c}_{stat}"] = p[c]
                stats[(key, c, stat)] = p[c]
        cnt = tx.pivot_table(index="signal_id", columns=col, values="d", aggfunc="size", observed=True).fillna(0)
        tot = cnt.sum(axis=1)
        for c in cnt.columns:
            out[f"{key}_{c}_n"] = cnt[c]
            out[f"{key}_{c}_frac"] = cnt[c] / tot
        # pairwise differences between categories
        for stat in ["mean", "median", "q75", "q25", "std"]:
            cats = [c for (k, c, s) in stats if k == key and s == stat]
            for a, b in itertools.combinations(cats, 2):
                out[f"{key}_{a}_minus_{b}_{stat}"] = stats[(key, a, stat)] - stats[(key, b, stat)]

    # each type's deviation from the signal's own overall mean, in std units
    g = tx.groupby("signal_id", observed=True).amt
    m, s = g.mean(), g.std()
    for t in TYPES:
        out[f"type_{t}_dev"] = (stats[("type", t, "mean")] - m) / s

    # the same relative signal inside recent windows
    for w in (30, 90):
        p = tx[tx.d < w].pivot_table(index="signal_id", columns="tranzaksiya_turi", values="amt",
                                     aggfunc="mean", observed=True)
        for c in p.columns:
            out[f"w{w}_type_{c}_mean"] = p[c]
        out[f"w{w}_bank_minus_naqd"] = p["bank_otkazmasi"] - p["naqd"]
        out[f"w{w}_bank_minus_karta"] = p["bank_otkazmasi"] - p["karta"]
    del tx; gc.collect()
    return pd.DataFrame(out).reindex(sig.signal_id)


FAMILY_SIZES = {}


def build_features(sig: pd.DataFrame, path: str) -> pd.DataFrame:
    """Run the three families (each re-reads the parquet to keep peak memory low)."""
    parts = []
    for fn in (features_windows, features_histograms, features_relative):
        t0 = time.time()
        parts.append(fn(sig, path))
        FAMILY_SIZES[fn.__name__] = int(parts[-1].shape[1])
        log(f"  {fn.__name__:22s} {parts[-1].shape[1]:4d} features  ({time.time() - t0:.0f}s)")
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X.replace([np.inf, -np.inf], np.nan).astype("float32")


# ----------------------------------------------------------------------------- models
def lr_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.01, max_iter=3000))


def rank01(p):
    return rankdata(p) / len(p)


def select_features(X, y):
    """Rank features by LightGBM gain under CV (with early stopping) and keep the top N."""
    imp = pd.Series(0.0, index=X.columns)
    oof = np.zeros(len(y))
    for tri, vai in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=5000, random_state=SEED, **LGB_PARAMS)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        imp += pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    log(f"all-feature LightGBM CV AUC = {roc_auc_score(y, oof):.4f}")
    imp = imp.sort_values(ascending=False)
    imp.rename("gain").to_csv(f"{OUT_DIR}/feature_importance.csv")
    top = list(imp.index[:N_TOP_FEATURES])
    with open(f"{OUT_DIR}/selected_features.txt", "w") as f:
        f.write("\n".join(top))
    log("top-10 features: " + ", ".join(top[:10]))
    return top, roc_auc_score(y, oof)


def cross_validate(X, y, cols, sig_ids, auc_all):
    """Honest CV of the final recipe on a different fold seed than the one used for selection."""
    o_lgb, o_lr = np.zeros(len(y)), np.zeros(len(y))
    for tri, vai in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 1).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=LGB_N_ESTIMATORS, random_state=SEED, **LGB_PARAMS)
        o_lgb[vai] = m.fit(X.iloc[tri][cols], y[tri]).predict_proba(X.iloc[vai][cols])[:, 1]
        o_lr[vai] = lr_model().fit(X.iloc[tri][cols], y[tri]).predict_proba(X.iloc[vai][cols])[:, 1]
    blend = BLEND_W_LGB * rank01(o_lgb) + (1 - BLEND_W_LGB) * rank01(o_lr)
    a_lgb, a_lr, a_bl = roc_auc_score(y, o_lgb), roc_auc_score(y, o_lr), roc_auc_score(y, blend)
    log(f"CV AUC  LightGBM={a_lgb:.4f}  LogReg={a_lr:.4f}  blend={a_bl:.4f}")
    pd.DataFrame({"signal_id": sig_ids, "eskalatsiya": y, "oof_blend": np.round(blend, 6),
                  "oof_lgb": np.round(o_lgb, 6), "oof_lr": np.round(o_lr, 6)}).to_csv(f"{OUT_DIR}/oof_predictions.csv", index=False)
    SUMMARY.update(cv_lgb=round(a_lgb, 4), cv_lr=round(a_lr, 4), cv_blend=round(a_bl, 4))
    with open(f"{OUT_DIR}/cv_report.txt", "w") as f:
        f.write(f"{N_FOLDS}-fold stratified CV, {len(cols)} selected features\n"
                f"all-feature LightGBM (early stopping): {auc_all:.4f}\n"
                f"LightGBM top-{len(cols)}:                {a_lgb:.4f}\n"
                f"LogReg top-{len(cols)}:                  {a_lr:.4f}\n"
                f"rank blend ({BLEND_W_LGB:.1f}/{1-BLEND_W_LGB:.1f}):                 {a_bl:.4f}\n")


def fit_predict(X, y, Xt, cols):
    p_lgb = np.zeros(len(Xt))
    for seed in range(N_SEEDS_FINAL):
        m = lgb.LGBMClassifier(n_estimators=LGB_N_ESTIMATORS, random_state=seed, **LGB_PARAMS)
        p_lgb += m.fit(X[cols], y).predict_proba(Xt[cols])[:, 1] / N_SEEDS_FINAL
    p_lr = lr_model().fit(X[cols], y).predict_proba(Xt[cols])[:, 1]
    return BLEND_W_LGB * rank01(p_lgb) + (1 - BLEND_W_LGB) * rank01(p_lr)


# ----------------------------------------------------------------------------- main
def main(quick: bool):
    import os; os.makedirs(OUT_DIR, exist_ok=True)
    global LOG_FILE
    LOG_FILE = f"{OUT_DIR}/train_log.txt"
    open(LOG_FILE, "w", encoding="utf-8").write(f"$ python src/train.py --team {TEAM_ID}\n")
    tr = pd.read_csv(f"{DATA_DIR}/train_signals.csv", parse_dates=["signal_sanasi"])
    te = pd.read_csv(f"{DATA_DIR}/test_signals.csv", parse_dates=["signal_sanasi"])
    y = tr.eskalatsiya.values
    log(f"train {tr.shape}, test {te.shape}, escalation rate {y.mean():.4f}")

    log("building train features")
    X = build_features(tr, f"{DATA_DIR}/train_transactions.parquet")
    log("building test features")
    Xt = build_features(te, f"{DATA_DIR}/test_transactions.parquet")
    Xt = Xt.reindex(columns=X.columns)
    log(f"feature matrix {X.shape}")

    cols, auc_all = select_features(X, y)
    SUMMARY.update(team=TEAM_ID, n_train=int(len(tr)), n_test=int(len(te)), pos_rate=round(float(y.mean()), 4),
                   n_features=int(X.shape[1]), families=dict(FAMILY_SIZES), n_selected=len(cols), cv_all_features=round(auc_all, 4),
                   n_folds=N_FOLDS, n_seeds=N_SEEDS_FINAL, blend_w_lgb=BLEND_W_LGB, n_estimators=LGB_N_ESTIMATORS,
                   lgb_params={k: v for k, v in LGB_PARAMS.items() if k != "verbose"}, lr_C=0.01,
                   burst_cutoff_minutes=round(BURST_CUTOFF_DAYS * 1440, 2))
    if not quick:
        cross_validate(X, y, cols, tr.signal_id.values, auc_all)

    log("fitting final models on all training data")
    pred = fit_predict(X, y, Xt, cols)

    sub = pd.DataFrame({"signal_id": te.signal_id.values, "ehtimollik": np.round(pred, 6)})
    assert len(sub) == len(te) and sub.signal_id.is_unique and sub.ehtimollik.between(0, 1).all()
    out = f"{OUT_DIR}/team_{TEAM_ID}.csv"
    sub.to_csv(out, index=False)
    log(f"wrote {out}  ({len(sub)} rows)")
    SUMMARY.update(submission=out, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    import json
    with open(f"{OUT_DIR}/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the extra CV pass")
    ap.add_argument("--data", default=DATA_DIR, help="folder with the competition files")
    ap.add_argument("--team", default=TEAM_ID, help="team id for the output file name")
    args = ap.parse_args()
    DATA_DIR, TEAM_ID = args.data, args.team
    main(args.quick)