"""Tugas 2 - Klasifikasi: deteksi fraud transaksi.

Alur:
  1. Audit label & fitur (kematangan label, kebocoran flag_review_manual, drift v15 / riwayat).
  2. Fitur perilaku hanya dari masa lalu (tanpa label).
  3. Validasi berbasis waktu dengan jeda (gap) untuk meniru keterlambatan konfirmasi label.
  4. Imbalance: LightGBM biasa, class weight, undersampling, + tuning threshold; baseline logistik.
  5. Pattern shift: validasi adversarial, PR-AUC per minggu, pembobotan recency.
  6. Kalibrasi (Platt vs isotonik) + keputusan berbasis biaya (p * nominal > Rp 25.000).
Keluaran: prediksi_klasifikasi.csv, hasil/klasifikasi.json, hasil/fig_klasifikasi_*.png
"""
import warnings

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DATA, HASIL, ROOT, SEED, Timer, save_json, set_seed

warnings.filterwarnings("ignore")
set_seed()
R = {}
COST_REVIEW = 25000.0

tr = pd.read_csv(DATA / "klasifikasi_train.csv.gz")
te = pd.read_csv(DATA / "klasifikasi_test.csv.gz")
tr["waktu"] = pd.to_datetime(tr.waktu)
te["waktu"] = pd.to_datetime(te.waktu)
R["n_train"], R["n_test"], R["fraud_rate_train"] = len(tr), len(te), float(tr.is_fraud.mean())

# ----------------------------------------------------------------------------
# 1. AUDIT
# ----------------------------------------------------------------------------
with Timer("audit"):
    A = {}
    wk = tr.assign(minggu=tr.waktu.dt.to_period("W").astype(str))
    A["fraud_rate_bulanan"] = tr.groupby(tr.waktu.dt.strftime("%Y-%m")).is_fraud.mean().round(5).to_dict()
    A["fraud_rate_mingguan_akhir"] = (wk[wk.waktu >= "2025-07-14"].groupby("minggu").is_fraud.mean()
                                      .round(5).to_dict())
    MATURE_END = pd.Timestamp("2025-08-11")  # transaksi < tanggal ini dianggap berlabel matang
    A["mature_end"] = "2025-08-10"
    A["rate_matang_Jul_Agu10"] = float(tr[(tr.waktu >= "2025-07-01") & (tr.waktu < MATURE_END)].is_fraud.mean())
    A["rate_1minggu_terakhir"] = float(tr[tr.waktu >= "2025-08-25"].is_fraud.mean())
    f1 = tr[tr.flag_review_manual == 1]
    A["flag_review_rate_fraud"] = float(f1.is_fraud.mean())
    A["flag_review_rate_nonflag"] = float(tr[tr.flag_review_manual == 0].is_fraud.mean())
    A["flag_review_auc"] = float(roc_auc_score(tr.is_fraud, tr.flag_review_manual))
    A["flag_review_share_train"] = float(tr.flag_review_manual.mean())
    A["flag_review_share_test"] = float(te.flag_review_manual.mean())
    A["fraud_dengan_flag_pct"] = float(tr[tr.is_fraud == 1].flag_review_manual.mean() * 100)
    mon = pd.concat([tr, te]).assign(bulan=lambda d: d.waktu.dt.strftime("%Y-%m"))
    A["v15_per_bulan"] = mon.groupby("bulan").v15.mean().round(2).to_dict()
    A["v15_auc"] = float(roc_auc_score(tr.is_fraud, tr.v15))
    A["menit_median_per_bulan"] = mon.groupby("bulan").menit_sejak_trx_terakhir.median().round(0).to_dict()
    A["hari_libur_share_train"] = float(tr.hari_libur.mean())
    A["hari_libur_share_test"] = float(te.hari_libur.mean())
    A["fraud_nominal_total_matang"] = float(tr[(tr.waktu < MATURE_END) & (tr.is_fraud == 1)].nominal.sum())
    A["n_matang"] = int((tr.waktu < MATURE_END).sum())
    A["pengguna_test_di_train_pct"] = float(te.id_pengguna.isin(tr.id_pengguna).mean() * 100)
    R["audit"] = A

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    w = wk.groupby("minggu").is_fraud.mean()
    ax[0].plot(range(len(w)), w.values)
    ax[0].axvline(list(w.index).index(wk[wk.waktu >= MATURE_END].minggu.iloc[0]), color="r", ls="--")
    ax[0].set_title("Fraud rate mingguan (garis merah: batas matang)")
    ax[0].set_xlabel("minggu ke-")
    mm = pd.Series(A["v15_per_bulan"])
    ax[1].plot(mm.index, mm.values, marker="o")
    ax[1].tick_params(axis="x", rotation=60)
    ax[1].set_title("v15: rata-rata per bulan")
    mm = pd.Series(A["menit_median_per_bulan"])
    ax[2].plot(mm.index, mm.values, marker="o", color="g")
    ax[2].tick_params(axis="x", rotation=60)
    ax[2].set_title("Median menit_sejak_trx_terakhir")
    plt.tight_layout()
    plt.savefig(HASIL / "fig_klasifikasi_audit.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------------
# 2. FITUR (hanya informasi masa lalu)
# ----------------------------------------------------------------------------
BIG = 10 ** 9


def rolling_count(codes, tsec, window):
    comb = codes.astype(np.int64) * BIG + tsec.astype(np.int64)
    order = np.argsort(comb, kind="mergesort")
    sc = comb[order]
    lo = np.searchsorted(sc, sc - window, side="left")
    cnt = np.empty(len(sc), dtype=np.int64)
    cnt[order] = np.arange(len(sc)) - lo
    return cnt


all_ = pd.concat([tr.assign(src="tr"), te.assign(src="te")], ignore_index=True)
all_["t"] = (all_.waktu - pd.Timestamp("2025-01-01")).dt.total_seconds().astype(np.int64)
all_ = all_.sort_values("t", kind="mergesort").reset_index(drop=True)

with Timer("fitur"):
    F = pd.DataFrame(index=all_.index)
    drop_cols = {"id_transaksi", "waktu", "id_pengguna", "id_merchant", "is_fraud", "flag_review_manual",
                 "v15", "hari_libur", "src", "t"}
    CAT = ["kanal", "kategori_merchant", "tipe_perangkat", "email_domain"]
    num_cols = [c for c in all_.columns if c not in drop_cols and c not in CAT]
    F[num_cols] = all_[num_cols].astype("float32")
    F["menit_sejak_trx_terakhir"] = np.log1p(all_.menit_sejak_trx_terakhir.clip(upper=10080)).astype("float32")
    F["rasio_nominal_rata2"] = all_.rasio_nominal_rata2.clip(upper=20).astype("float32")
    F["log_nominal"] = np.log1p(all_.nominal).astype("float32")
    F["jam_sin"] = np.sin(2 * np.pi * all_.jam / 24).astype("float32")
    F["jam_cos"] = np.cos(2 * np.pi * all_.jam / 24).astype("float32")
    F["dow"] = all_.waktu.dt.dayofweek.astype("float32")
    for c in CAT:
        F[c] = pd.Categorical(all_[c])

    # --- perilaku pengguna / merchant dari data masa lalu saja ---
    ucode = all_.id_pengguna.astype("category").cat.codes.values
    mcode = all_.id_merchant.astype("category").cat.codes.values
    pair = all_.id_pengguna.astype(str) + "|" + all_.id_merchant.astype(str)
    pcode = pair.astype("category").cat.codes.values
    t = all_.t.values
    nom = all_.nominal.values

    def win_stats(codes, window, name):
        """Hitungan dan rata-rata nominal pada jendela `window` detik SEBELUM transaksi (stasioner terhadap waktu)."""
        comb = codes.astype(np.int64) * BIG + t.astype(np.int64)
        order = np.argsort(comb, kind="mergesort")
        sc = comb[order]
        lo = np.searchsorted(sc, sc - window, side="left")
        pos = np.arange(len(sc))
        cs = np.r_[0.0, np.cumsum(nom[order])]  # prefiks eksklusif
        cnt_s = pos - lo
        sum_s = cs[pos] - cs[lo]
        cnt = np.empty(len(sc))
        sm = np.empty(len(sc))
        cnt[order], sm[order] = cnt_s, sum_s
        F[f"{name}_cnt_30h"] = cnt.astype("float32")
        mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        F[f"{name}_rasio_mean_30h"] = (nom / mean).clip(max=50).astype("float32")

    D30 = 30 * 86400
    win_stats(ucode, D30, "u")
    win_stats(mcode, D30, "m")
    win_stats(pcode, D30, "um")
    F["m_cnt_1j"] = rolling_count(mcode, t, 3600).astype("float32")
    F["m_cnt_24j"] = rolling_count(mcode, t, 86400).astype("float32")
    F["u_cnt_10m"] = rolling_count(ucode, t, 600).astype("float32")
    A["fitur_perilaku"] = ["u/m/um_cnt_30h", "u/m/um_rasio_mean_30h", "m_cnt_1j", "m_cnt_24j", "u_cnt_10m"]
    # dua fitur yang dihitung dari riwayat bergeser karena data dimulai 1 Jan: cek drift
    R["n_fitur"] = int(F.shape[1])

is_tr = (all_.src == "tr").values
y_all = all_.is_fraud.fillna(0).values.astype(int)
tw = all_.waktu


def idx(mask):
    return np.where(mask)[0]


MATURE = idx(is_tr & (tw < MATURE_END).values)


def period(a, b):
    return idx(is_tr & (tw >= a).values & (tw < b).values)


# ----------------------------------------------------------------------------
# 3. SKEMA VALIDASI (gap 21 hari meniru keterlambatan label)
# ----------------------------------------------------------------------------
FOLDS = {
    "A": (period("2025-01-01", "2025-05-01"), period("2025-05-22", "2025-07-01")),
    "B": (period("2025-01-01", "2025-06-16"), period("2025-07-07", "2025-08-11")),
}
R["folds"] = {"A": "latih 1 Jan-30 Apr, jeda 21 hari, validasi 22 Mei-30 Jun",
              "B": "latih 1 Jan-15 Jun, jeda 21 hari, validasi 7 Jul-10 Agu (label matang)"}


def cost_of(y, nominal, flag):
    return float(nominal[(y == 1) & (flag == 0)].sum() + COST_REVIEW * flag.sum())


def lgb_fit(ix, params, n_est=None, valix=None, weight=None, spw=1.0):
    p = dict(objective="binary", learning_rate=0.05, n_estimators=n_est or 1500, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, random_state=SEED, n_jobs=1, verbose=-1,
             scale_pos_weight=spw, **params)
    m = lgb.LGBMClassifier(**p)
    kw = {}
    if valix is not None:
        kw = dict(eval_set=[(F.iloc[valix], y_all[valix])], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    m.fit(F.iloc[ix], y_all[ix], sample_weight=weight, **kw)
    return m


GRID = [dict(num_leaves=15, min_child_samples=100, reg_lambda=5.0),
        dict(num_leaves=63, min_child_samples=50, reg_lambda=10.0)]

with Timer("tuning fold A"):
    a_tr, a_va = FOLDS["A"]
    tune = []
    for i, p in enumerate(GRID):
        m = lgb_fit(a_tr, p, valix=a_va)
        pr = m.predict_proba(F.iloc[a_va])[:, 1]
        r = dict(config=i, pr_auc=float(average_precision_score(y_all[a_va], pr)),
                 roc_auc=float(roc_auc_score(y_all[a_va], pr)), best_iter=int(m.best_iteration_), **p)
        tune.append(r)
        print(r, flush=True)
    best = max(tune, key=lambda r: r["pr_auc"])
    BEST, NIT = GRID[best["config"]], max(100, int(best["best_iter"] * 1.15))
    R["tuning"] = tune
    R["best_config"] = dict(BEST, n_estimators=NIT)

# ----------------------------------------------------------------------------
# 4. IMBALANCE (Fold B)
# ----------------------------------------------------------------------------
b_tr, b_va = FOLDS["B"]
yv = y_all[b_va]
nom_v = all_.nominal.values[b_va]
R["fold_B"] = dict(n_train=int(len(b_tr)), n_val=int(len(b_va)), fraud_train=int(y_all[b_tr].sum()),
                   fraud_val=int(yv.sum()))
scores = {}

with Timer("imbalance"):
    # (0) baseline logistik
    num_f = [c for c in F.columns if str(F[c].dtype) != "category"]
    cat_f = CAT
    pre = ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_f),
        ("c", OneHotEncoder(handle_unknown="ignore"), cat_f)])
    lr = make_pipeline(pre, LogisticRegression(C=0.3, max_iter=300, n_jobs=1))
    lr.fit(F.iloc[b_tr], y_all[b_tr])
    scores["logistik"] = lr.predict_proba(F.iloc[b_va])[:, 1]
    # (1) LightGBM tanpa penanganan
    m_plain = lgb_fit(b_tr, BEST, n_est=NIT)
    scores["lgbm_biasa"] = m_plain.predict_proba(F.iloc[b_va])[:, 1]
    # (2) class weight
    pos = y_all[b_tr].mean()
    spw = float(np.sqrt((1 - pos) / pos))
    m_w = lgb_fit(b_tr, BEST, n_est=NIT, spw=spw)
    scores["lgbm_class_weight"] = m_w.predict_proba(F.iloc[b_va])[:, 1]
    R["spw"] = spw
    # (3) undersampling negatif (simpan 10 persen), koreksi probabilitas
    rng = np.random.default_rng(SEED)
    keep = b_tr[(y_all[b_tr] == 1) | (rng.random(len(b_tr)) < 0.10)]
    m_u = lgb_fit(keep, BEST, n_est=NIT)
    ps = m_u.predict_proba(F.iloc[b_va])[:, 1]
    beta = 0.10
    scores["lgbm_undersampling"] = beta * ps / (1 - ps + beta * ps)
    # (4) recency weighting (pattern shift)
    age = (pd.Timestamp("2025-06-16") - tw.iloc[b_tr]).dt.days.values
    w = 0.5 ** (age / 45.0)
    m_r = lgb_fit(b_tr, BEST, n_est=NIT, weight=w)
    scores["lgbm_recency"] = m_r.predict_proba(F.iloc[b_va])[:, 1]

    cmp = {}
    for k, s in scores.items():
        cmp[k] = dict(pr_auc=float(average_precision_score(yv, s)), roc_auc=float(roc_auc_score(yv, s)))
        print(k, cmp[k], flush=True)
    R["perbandingan_foldB"] = cmp

# ----------------------------------------------------------------------------
# 5. PATTERN SHIFT
# ----------------------------------------------------------------------------
with Timer("pattern shift"):
    best_name = max([k for k in scores if k != "logistik"], key=lambda k: cmp[k]["pr_auc"])
    R["metode_terbaik_foldB"] = best_name
    sb = scores[best_name]
    wkv = tw.iloc[b_va].dt.to_period("W").astype(str).values
    per_week = {}
    for wk_ in np.unique(wkv):
        mk = wkv == wk_
        if yv[mk].sum() >= 5:
            per_week[wk_] = dict(n=int(mk.sum()), fraud=int(yv[mk].sum()),
                                 pr_auc=float(average_precision_score(yv[mk], sb[mk])))
    R["pr_auc_mingguan_foldB"] = per_week

    # fraud share per kanal per bulan (apakah pola berubah?)
    f = tr[tr.waktu < MATURE_END]
    f = f[f.is_fraud == 1]
    mix = pd.crosstab(f.waktu.dt.strftime("%Y-%m"), f.kanal, normalize="index").round(3)
    R["fraud_mix_kanal"] = mix.to_dict()
    mixk = pd.crosstab(f.waktu.dt.strftime("%Y-%m"), f.kategori_merchant, normalize="index")
    R["fraud_mix_kategori_top"] = mixk[mixk.mean().sort_values(ascending=False).head(5).index].round(3).to_dict()

    # validasi adversarial: pisahkan Jul-Agu(latih) vs test (kovariat saja)
    late = idx(is_tr & (tw >= "2025-07-01").values & (tw < MATURE_END).values)
    tst = idx(~is_tr)
    rs = np.random.default_rng(SEED)
    sub_l, sub_t = rs.choice(late, 40000, replace=False), rs.choice(tst, 40000, replace=False)
    ix = np.r_[sub_l, sub_t]
    ya = np.r_[np.zeros(40000), np.ones(40000)].astype(int)
    perm = rs.permutation(len(ix))
    ix, ya = ix[perm], ya[perm]
    half = len(ix) // 2
    adv = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=1, verbose=-1,
                             random_state=SEED).fit(F.iloc[ix[:half]], ya[:half])
    R["adversarial_auc"] = float(roc_auc_score(ya[half:], adv.predict_proba(F.iloc[ix[half:]])[:, 1]))
    R["adversarial_top"] = (pd.Series(adv.booster_.feature_importance("gain"), index=F.columns)
                            .sort_values(ascending=False).head(6).round(0).to_dict())
    print("adversarial", R["adversarial_auc"], R["adversarial_top"], flush=True)

# ----------------------------------------------------------------------------
# 5b. POLA BARU: LONJAKAN FRAUD QRIS DI MERCHANT MUDA (mulai akhir Juli)
# ----------------------------------------------------------------------------
with Timer("pola baru qris"):
    mat_df = tr[tr.waktu < MATURE_END]
    qf = mat_df[(mat_df.kanal == "qris") & (mat_df.is_fraud == 1)]
    new_f = qf[qf.waktu >= "2025-07-28"]
    old_f = qf[qf.waktu < "2025-07-28"]
    wkq = mat_df[(mat_df.kanal == "qris") & (mat_df.waktu >= "2025-07-07")]
    S = dict(
        qris_mingguan=wkq.groupby(wkq.waktu.dt.to_period("W").astype(str)).is_fraud.agg(["size", "sum"]).rename(
            columns={"size": "n", "sum": "fraud"}).to_dict(orient="index"),
        n_fraud_qris_baru=int(len(new_f)), n_fraud_qris_lama=int(len(old_f)),
        n_merchant_baru=int(new_f.id_merchant.nunique()),
        umur_merchant_median_baru=float(new_f.umur_merchant_hari.median()),
        umur_merchant_median_lama=float(old_f.umur_merchant_hari.median()),
        nominal_median_baru=float(new_f.nominal.median()), nominal_median_lama=float(old_f.nominal.median()),
        kategori_baru=new_f.kategori_merchant.value_counts(normalize=True).head(3).round(2).to_dict(),
        kategori_lama=old_f.kategori_merchant.value_counts(normalize=True).head(3).round(2).to_dict())
    ring = set(new_f.id_merchant)
    S["trx_test_di_merchant_ini"] = int(te.id_merchant.isin(ring).sum())
    S["fraud_rate_merchant_ini_matang"] = float(mat_df[mat_df.id_merchant.isin(ring)].is_fraud.mean())
    S["fraud_rate_merchant_muda_30h_sebelum"] = float(mat_df[(mat_df.waktu < "2025-07-28") & (mat_df.umur_merchant_hari <= 30)].is_fraud.mean())
    S["fraud_rate_merchant_muda_30h_sesudah"] = float(mat_df[(mat_df.waktu >= "2025-07-28") & (mat_df.umur_merchant_hari <= 30)].is_fraud.mean())
    R["pola_baru_qris"] = S
    print(S, flush=True)

    # Adaptasi: latih sampai 4 Agu (termasuk minggu pertama pola baru), validasi 5-10 Agu.
    ad_tr, ad_va = period("2025-01-01", "2025-08-05"), period("2025-08-05", "2025-08-11")
    yav, nav = y_all[ad_va], all_.nominal.values[ad_va]
    ref = pd.Timestamp("2025-08-05")
    ageA = (ref - tw.iloc[ad_tr]).dt.days.values.astype(float)
    variants = {
        "tanpa_bobot": None,
        "bobot_x5_sejak_28Jul": 1.0 + 4.0 * (tw.iloc[ad_tr] >= "2025-07-28").values,
        "recency_paruh_14h": 0.5 ** (np.maximum(ageA, 0) / 14.0) + 0.05,
        "recency_paruh_45h": 0.5 ** (np.maximum(ageA, 0) / 45.0),
    }
    adapt = {}
    for nm_, w_ in variants.items():
        m_ = lgb_fit(ad_tr, BEST, n_est=NIT, weight=w_)
        pr_ = m_.predict_proba(F.iloc[ad_va])[:, 1]
        qm = (all_.kanal.values[ad_va] == "qris")
        adapt[nm_] = dict(pr_auc=float(average_precision_score(yav, pr_)),
                          pr_auc_qris=float(average_precision_score(yav[qm], pr_[qm])) if yav[qm].sum() > 0 else None,
                          fraud=int(yav.sum()), fraud_qris=int(yav[qm].sum()),
                          biaya_aturan=cost_of(yav, nav, (pr_ * nav > COST_REVIEW).astype(int)))
        print(nm_, adapt[nm_], flush=True)
    R["adaptasi_pola_baru"] = adapt
    ADAPT = max(adapt, key=lambda k: adapt[k]["pr_auc"])
    R["adaptasi_terpilih"] = ADAPT

# ----------------------------------------------------------------------------
# 6. KALIBRASI + KEPUTUSAN BERBASIS BIAYA
# ----------------------------------------------------------------------------
def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(y=("y", "mean"), p=("p", "mean"),
                                                                                n=("y", "size"))
    return float((g.n * (g.y - g.p).abs()).sum() / g.n.sum())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


with Timer("kalibrasi"):
    order = np.argsort(tw.iloc[b_va].values)
    ntot = len(order)
    cal_i, chk_i = order[: ntot // 2], order[ntot // 2:]  # paruh awal kalibrasi, paruh akhir uji
    s_raw = scores[best_name]
    platt = LogisticRegression(C=1e6).fit(logit(s_raw[cal_i]).reshape(-1, 1), yv[cal_i])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_raw[cal_i], yv[cal_i])
    p_platt = platt.predict_proba(logit(s_raw[chk_i]).reshape(-1, 1))[:, 1]
    p_iso = iso.predict(s_raw[chk_i])
    R["kalibrasi"] = {
        "mentah": dict(brier=float(brier_score_loss(yv[chk_i], s_raw[chk_i])), ece=ece(yv[chk_i], s_raw[chk_i]),
                       mean_p=float(s_raw[chk_i].mean())),
        "platt": dict(brier=float(brier_score_loss(yv[chk_i], p_platt)), ece=ece(yv[chk_i], p_platt),
                      mean_p=float(p_platt.mean())),
        "isotonik": dict(brier=float(brier_score_loss(yv[chk_i], p_iso)), ece=ece(yv[chk_i], p_iso),
                         mean_p=float(p_iso.mean())),
        "rate_aktual": float(yv[chk_i].mean())}
    print(R["kalibrasi"], flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, p in [("mentah", s_raw[chk_i]), ("Platt", p_platt), ("isotonik", p_iso)]:
        q = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"y": yv[chk_i], "p": p, "q": q}).groupby("q", observed=True).mean()
        ax[0].plot(g.p, g.y, marker="o", label=name)
    mx = max(0.05, float(ax[0].get_xlim()[1]))
    ax[0].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[0].set_xscale("symlog", linthresh=1e-3)
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_title("Reliability (paruh akhir validasi B)")
    ax[0].set_xlabel("prob prediksi")
    ax[0].set_ylabel("frekuensi aktual")
    ax[0].legend()

    # kebijakan keputusan, dievaluasi pada paruh akhir (kalibrasi di paruh awal)
    yc, nc = yv[chk_i], nom_v[chk_i]
    pol = {}
    pol["tidak_tandai_apa_pun"] = cost_of(yc, nc, np.zeros(len(yc), int))
    pol["tandai_semua"] = cost_of(yc, nc, np.ones(len(yc), int))
    pol["aturan_biaya_platt (p*nominal>25rb)"] = cost_of(yc, nc, (p_platt * nc > COST_REVIEW).astype(int))
    pol["aturan_biaya_isotonik"] = cost_of(yc, nc, (p_iso * nc > COST_REVIEW).astype(int))
    pol["aturan_biaya_mentah (tanpa kalibrasi)"] = cost_of(yc, nc, (s_raw[chk_i] * nc > COST_REVIEW).astype(int))
    # threshold global pada p yang dituning di paruh awal
    pc = platt.predict_proba(logit(s_raw[cal_i]).reshape(-1, 1))[:, 1]
    grid = np.unique(np.quantile(pc, np.linspace(0.5, 0.9995, 200)))
    costs = [cost_of(yv[cal_i], nom_v[cal_i], (pc >= g).astype(int)) for g in grid]
    thr = float(grid[int(np.argmin(costs))])
    pol["threshold_global_p (dituning paruh awal)"] = cost_of(yc, nc, (p_platt >= thr).astype(int))
    R["threshold_global"] = thr
    R["kebijakan_biaya"] = pol
    R["kebijakan_ringkas"] = dict(n=int(len(yc)), fraud=int(yc.sum()), nominal_fraud=float(nc[yc == 1].sum()),
                                  n_flag_aturan=int((p_platt * nc > COST_REVIEW).sum()),
                                  recall_aturan=float(((p_platt * nc > COST_REVIEW) & (yc == 1)).sum() / max(1, yc.sum())))
    for k, v in pol.items():
        print(f"  {k}: Rp {v:,.0f}", flush=True)
    names = [k.split(" (")[0] for k in pol]
    ax[1].barh(names, [v / 1e6 for v in pol.values()], color="#3b6fb6")
    ax[1].set_xlabel("total biaya (juta Rp)")
    ax[1].set_title("Total biaya per kebijakan")
    ax[1].tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(HASIL / "fig_klasifikasi_kalibrasi.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------------
# 7. ANALISIS ERROR
# ----------------------------------------------------------------------------
with Timer("analisis"):
    flag_v = (platt.predict_proba(logit(s_raw).reshape(-1, 1))[:, 1] * nom_v > COST_REVIEW).astype(int)
    dfv = pd.DataFrame({"y": yv, "s": s_raw, "flag": flag_v, "nom": nom_v,
                        "kanal": all_.kanal.values[b_va], "kat": all_.kategori_merchant.values[b_va]})

    def seg(col):
        out = {}
        for k, g in dfv.groupby(col):
            if g.y.sum() >= 3:
                out[k] = dict(n=int(len(g)), fraud=int(g.y.sum()),
                              pr_auc=float(average_precision_score(g.y, g.s)),
                              recall=float(g[g.y == 1].flag.mean()),
                              biaya_miss=float(g[(g.y == 1) & (g.flag == 0)].nom.sum()),
                              n_flag=int(g.flag.sum()))
        return out

    R["error_kanal"] = seg("kanal")
    R["error_kategori"] = seg("kat")
    # pola fraud yang terlewat vs tertangkap
    Xv = F.iloc[b_va].reset_index(drop=True)
    fr = dfv.y.values == 1
    cau, mis = fr & (dfv.flag.values == 1), fr & (dfv.flag.values == 0)
    num_cols_v = [c for c in Xv.columns if str(Xv[c].dtype) != "category"]
    sd = Xv[num_cols_v].std().replace(0, np.nan)
    diff = ((Xv.loc[mis, num_cols_v].mean() - Xv.loc[cau, num_cols_v].mean()) / sd).abs().sort_values(
        ascending=False)
    R["miss_vs_caught_top"] = {k: dict(missed=float(Xv.loc[mis, k].mean()), caught=float(Xv.loc[cau, k].mean()))
                               for k in diff.head(6).index}
    R["n_missed"], R["n_caught"] = int(mis.sum()), int(cau.sum())
    R["median_nominal_missed"] = float(dfv.nom[mis].median())
    R["median_nominal_caught"] = float(dfv.nom[cau].median())
    # fraud nominal kecil: biaya tinjau > kerugian
    R["fraud_nominal_<25rb_pct"] = float((dfv.nom[fr] < COST_REVIEW).mean() * 100)

    imp = pd.Series(m_plain.booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    R["importance_top"] = (imp.head(15) / imp.sum()).round(4).to_dict()

# ----------------------------------------------------------------------------
# 8. MODEL FINAL + PREDIKSI TEST
# ----------------------------------------------------------------------------
with Timer("final"):
    final_ix = MATURE
    ref_f = pd.Timestamp("2025-08-11")
    ageF = (ref_f - tw.iloc[final_ix]).dt.days.values.astype(float)
    w_f = {"tanpa_bobot": None,
           "bobot_x5_sejak_28Jul": 1.0 + 4.0 * (tw.iloc[final_ix] >= "2025-07-28").values,
           "recency_paruh_14h": 0.5 ** (np.maximum(ageF, 0) / 14.0) + 0.05,
           "recency_paruh_45h": 0.5 ** (np.maximum(ageF, 0) / 45.0)}[ADAPT]
    R["final_adaptasi"] = ADAPT
    spw_f = spw if best_name == "lgbm_class_weight" else 1.0
    m_f = lgb_fit(final_ix, BEST, n_est=int(NIT * 1.1), weight=w_f, spw=spw_f)
    te_ix = idx(~is_tr)
    s_te = m_f.predict_proba(F.iloc[te_ix])[:, 1]
    p_te = platt.predict_proba(logit(s_te).reshape(-1, 1))[:, 1]
    nom_te = all_.nominal.values[te_ix]
    flag_te = (p_te * nom_te > COST_REVIEW).astype(int)
    sub = pd.DataFrame({"id_transaksi": all_.id_transaksi.values[te_ix], "prob": np.round(p_te, 6),
                        "tandai": flag_te})
    sub = te[["id_transaksi"]].merge(sub, on="id_transaksi", how="left")
    assert len(sub) == len(te) and sub.id_transaksi.is_unique and sub.prob.between(0, 1).all()
    assert sub.tandai.isin([0, 1]).all()
    sub.to_csv(ROOT / "prediksi_klasifikasi.csv", index=False)
    ringm = te.set_index("id_transaksi").id_merchant.isin(ring).reindex(sub.id_transaksi).values
    R["test_merchant_pola_baru"] = dict(n=int(ringm.sum()), n_flag=int(sub.tandai.values[ringm].sum()),
                                        mean_prob=float(sub.prob.values[ringm].mean()))
    R["test"] = dict(mean_prob=float(sub.prob.mean()), n_flag=int(sub.tandai.sum()),
                     flag_rate=float(sub.tandai.mean()), mean_raw=float(s_te.mean()))
    print(R["test"], flush=True)

save_json(R, "klasifikasi.json")
print("SELESAI klasifikasi")
