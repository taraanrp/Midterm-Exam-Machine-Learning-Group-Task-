"""Tugas 1 - Regresi: estimasi durasi pengiriman (P50 dan P90).

Alur:
  1. Audit data (kebocoran, drift, outlier, kategori baru).
  2. Validasi berbasis waktu (Fold A untuk tuning, Fold B untuk pelaporan/kalibrasi).
  3. Model: baseline grup, Ridge (target log) + kuantil residual, LightGBM kuantil (target log).
  4. Kalibrasi P90 dengan conformal (Mondrian per layanan).
  5. Analisis error + SHAP, lalu prediksi test.
Keluaran: prediksi_regresi.csv, hasil/regresi.json, hasil/fig_regresi_*.png
"""
import warnings

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from common import DATA, HASIL, ROOT, SEED, Timer, pinball, save_json, set_seed

warnings.filterwarnings("ignore")
set_seed()
R = {}  # semua angka untuk laporan

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
tr = pd.read_csv(DATA / "regresi_train.csv.gz")
te = pd.read_csv(DATA / "regresi_test.csv.gz")
tr["tgl"] = pd.to_datetime(tr.tgl_pickup)
te["tgl"] = pd.to_datetime(te.tgl_pickup)
R["n_train"], R["n_test"] = len(tr), len(te)

F_COLS = [f"f{i:02d}" for i in range(1, 51)]
BASE_NUM = ["jarak_km", "berat_kg", "volume_liter", "jumlah_item", "nilai_barang", "asuransi", "cod",
            "alamat_skor", "pengalaman_kurir_bulan", "rating_penjual", "umur_akun_penjual_hari",
            "volume_penjual_30h", "antrean_hub_asal", "indeks_kemacetan", "hujan_asal_mm", "hujan_tujuan_mm"]
LOW_CARD = ["layanan", "hub_asal", "pulau_tujuan"]
CATS = {c: sorted(set(tr[c]) | set(te[c])) for c in LOW_CARD}
CATS["mitra_kurir"] = sorted(set(tr.mitra_kurir))  # sengaja hanya kategori yang ada di train
HIGH_CARD = ["kota_tujuan", "kota_asal"]

# ----------------------------------------------------------------------------
# 1. AUDIT DATA
# ----------------------------------------------------------------------------
with Timer("audit"):
    A = {}
    A["jumlah_scan_mean_train"] = float(tr.jumlah_scan.mean())
    A["jumlah_scan_mean_test"] = float(te.jumlah_scan.mean())
    A["corr_scan_durasi"] = float(tr.jumlah_scan.corr(tr.durasi_jam, method="spearman"))
    # f30 bergeser linear terhadap waktu dan tidak berkorelasi dengan target
    mon = pd.concat([tr, te]).assign(bulan=lambda d: d.tgl.dt.strftime("%Y-%m"))
    A["f30_per_bulan"] = mon.groupby("bulan").f30.mean().round(3).to_dict()
    A["corr_f30_durasi"] = float(tr.f30.corr(tr.durasi_jam, method="spearman"))
    A["f_drift_top"] = (((te[F_COLS].mean() - tr[F_COLS].mean()) / tr[F_COLS].std()).abs()
                        .sort_values(ascending=False).head(4).round(3).to_dict())
    A["antrean_mean_train"] = float(tr.antrean_hub_asal.mean())
    A["antrean_mean_test"] = float(te.antrean_hub_asal.mean())
    A["outlier_gt500"] = int((tr.durasi_jam > 500).sum())
    A["outlier_gt1000"] = int((tr.durasi_jam > 1000).sum())
    A["durasi_p99"] = float(tr.durasi_jam.quantile(0.99))
    A["durasi_max"] = float(tr.durasi_jam.max())
    A["jarak_nol"] = int((tr.jarak_km <= 0).sum())
    A["berat_missing_pct"] = float(tr.berat_kg.isna().mean() * 100)
    A["mitra_baru_test"] = sorted(set(te.mitra_kurir) - set(tr.mitra_kurir))
    A["mitra_baru_test_pct"] = float((~te.mitra_kurir.isin(tr.mitra_kurir)).mean() * 100)
    A["kota_tujuan_baru_test"] = int((~te.kota_tujuan.isin(tr.kota_tujuan)).sum())
    A["kota_tujuan_nunique"] = int(tr.kota_tujuan.nunique())
    A["duplikat_id"] = int(tr.id_paket.duplicated().sum())
    R["audit"] = A

# Pembersihan target: durasi > 1000 jam (> 4x P99) dibuang HANYA dari data latih.
CLEAN_THR = 1000.0
tr["ok"] = tr.durasi_jam <= CLEAN_THR


# ----------------------------------------------------------------------------
# 2. FITUR
# ----------------------------------------------------------------------------
def build(df, cats=CATS, use_scan=False, use_f30=False, mask_mitra_p=0.0, rng=None):
    X = df[BASE_NUM].copy()
    X["jarak_km"] = X["jarak_km"].where(X["jarak_km"] > 0)  # jarak 0 tidak masuk akal -> NaN
    f_use = [c for c in F_COLS if use_f30 or c != "f30"]
    X[f_use] = df[f_use]
    if use_scan:
        X["jumlah_scan"] = df["jumlah_scan"]
    X["jam"] = df.jam_pickup
    X["dow"] = df.tgl.dt.dayofweek
    X["dom"] = df.tgl.dt.day
    X["berat_missing"] = df.berat_kg.isna().astype(int)
    X["kepadatan"] = df.berat_kg / df.volume_liter
    for c in LOW_CARD + ["mitra_kurir"]:
        v = df[c].copy()
        if c == "mitra_kurir" and mask_mitra_p > 0:
            v = v.mask(rng.random(len(v)) < mask_mitra_p)
        X[c] = pd.Categorical(v, categories=cats[c])
    return X


def target_encode(Xtr, ytr, df_tr, others, smooth=20, n_splits=5):
    """Target encoding kota (kardinalitas tinggi): out-of-fold untuk train, pemetaan penuh untuk lainnya.
    Kota yang tidak pernah terlihat -> rata-rata global."""
    prior = ytr.mean()
    kf = KFold(n_splits, shuffle=True, random_state=SEED)
    for c in HIGH_CARD:
        oof = np.full(len(df_tr), np.nan)
        keys = df_tr[c].values
        for a, b in kf.split(keys):
            s = pd.DataFrame({"k": keys[a], "y": ytr[a]}).groupby("k").y.agg(["sum", "count"])
            m = (s["sum"] + smooth * prior) / (s["count"] + smooth)
            oof[b] = pd.Series(keys[b]).map(m).fillna(prior).values
        Xtr["te_" + c] = oof
        s = pd.DataFrame({"k": keys, "y": ytr}).groupby("k").y.agg(["sum", "count"])
        m = (s["sum"] + smooth * prior) / (s["count"] + smooth)
        for Xo, dfo in others:
            Xo["te_" + c] = dfo[c].map(m).fillna(prior).values
    return Xtr, [o[0] for o in others]


def prepare(df_tr, df_others, use_scan=False, use_f30=False, mask_p=0.0):
    rng = np.random.default_rng(SEED)
    ytr = np.log(df_tr.durasi_jam.values)
    Xtr = build(df_tr, use_scan=use_scan, use_f30=use_f30, mask_mitra_p=mask_p, rng=rng)
    Xo = [build(d, use_scan=use_scan, use_f30=use_f30) for d in df_others]
    Xtr, Xo = target_encode(Xtr, ytr, df_tr, list(zip(Xo, df_others)))
    return Xtr, ytr, Xo


# ----------------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------------
GRID = [
    dict(num_leaves=31, min_child_samples=200, reg_lambda=5.0),
    dict(num_leaves=63, min_child_samples=100, reg_lambda=5.0),
    dict(num_leaves=127, min_child_samples=50, reg_lambda=10.0),
]
LR = 0.1


def fit_lgb(X, y, alpha, params, n_est=None, Xv=None, yv=None):
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=n_est or 1500, learning_rate=LR,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED,
                          n_jobs=1, verbose=-1, **params)
    if Xv is not None:
        m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="quantile",
              callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(X, y)
    return m


def ridge_pipeline(Xtr):
    num = [c for c in Xtr.columns if str(Xtr[c].dtype) != "category"]
    cat = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    pre = ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, Ridge(alpha=10.0))


def ridge_fit_predict(Xtr, ytr, layanan_tr, Xs, layanan_s, qs=(0.5, 0.9)):
    """Ridge pada log(durasi); kuantil = prediksi + kuantil residual per layanan (target log)."""
    Xtr = Xtr.copy()
    Xs = [x.copy() for x in Xs]
    # fitur log untuk variabel miring
    for X in [Xtr] + Xs:
        for c in ["jarak_km", "berat_kg", "volume_liter", "nilai_barang", "antrean_hub_asal"]:
            X[c] = np.log1p(X[c])
        X["kepadatan"] = np.log1p(X["kepadatan"])
    pipe = ridge_pipeline(Xtr).fit(Xtr, ytr)
    res = ytr - pipe.predict(Xtr)
    out = []
    for X, lay in zip(Xs, layanan_s):
        mu = pipe.predict(X)
        d = {}
        for q in qs:
            off = pd.Series(res).groupby(layanan_tr.values).quantile(q)
            d[q] = mu + lay.map(off).values
        out.append(d)
    return out


def baseline_quantiles(df_tr, df_s, qs=(0.5, 0.9)):
    """Baseline: kuantil empiris durasi per (layanan x pulau_tujuan)."""
    g = df_tr.groupby(["layanan", "pulau_tujuan"]).durasi_jam
    out = {}
    for q in qs:
        m = g.quantile(q).rename("v").reset_index()
        default = df_tr.durasi_jam.quantile(q)
        merged = df_s[["layanan", "pulau_tujuan"]].merge(m, on=["layanan", "pulau_tujuan"], how="left")
        out[q] = merged["v"].fillna(default).values
    return out


def metrics(y, p50, p90):
    return dict(pin50=pinball(y, p50, 0.5), pin90=pinball(y, p90, 0.9), cov90=float(np.mean(y <= p90)),
                pin_avg=(pinball(y, p50, 0.5) + pinball(y, p90, 0.9)) / 2)


# ----------------------------------------------------------------------------
# 4. VALIDASI WAKTU
# ----------------------------------------------------------------------------
folds = {
    "A": (tr.tgl < "2025-05-01", (tr.tgl >= "2025-05-01") & (tr.tgl < "2025-07-01")),  # tuning
    "B": (tr.tgl < "2025-07-01", tr.tgl >= "2025-07-01"),  # pelaporan + kalibrasi
}
R["folds"] = {"A": "latih Jan-Apr, validasi Mei-Jun (tuning)", "B": "latih Jan-Jun, validasi Jul-Agu (pelaporan)"}

# ---- Fold A: tuning --------------------------------------------------------
with Timer("tuning fold A"):
    mtr, mva = folds["A"]
    dtr, dva = tr[mtr & tr.ok], tr[mva]
    Xa, ya, (Xva,) = prepare(dtr, [dva])
    yva_log = np.log(dva.durasi_jam.values)
    tune = []
    for i, p in enumerate(GRID):
        m5 = fit_lgb(Xa, ya, 0.5, p, Xv=Xva, yv=yva_log)
        m9 = fit_lgb(Xa, ya, 0.9, p, Xv=Xva, yv=yva_log)
        r = metrics(dva.durasi_jam.values, np.exp(m5.predict(Xva)), np.exp(m9.predict(Xva)))
        r.update(config=i, it50=int(m5.best_iteration_), it90=int(m9.best_iteration_), **p)
        tune.append(r)
        print(r, flush=True)
    best = min(tune, key=lambda r: r["pin_avg"])
    BEST = GRID[best["config"]]
    IT50, IT90 = best["it50"], best["it90"]
    R["tuning"] = tune
    R["best_config"] = dict(BEST, it50=IT50, it90=IT90)

# ---- Fold B: pelaporan ------------------------------------------------------
with Timer("fold B"):
    mtr, mva = folds["B"]
    dtr, dva = tr[mtr & tr.ok], tr[mva].copy()
    yv = dva.durasi_jam.values
    Xb, yb, (Xvb,) = prepare(dtr, [dva], mask_p=0.04)
    models = {}
    for q, it in [(0.5, IT50), (0.9, IT90)]:
        models[q] = fit_lgb(Xb, yb, q, BEST, n_est=int(it * 1.1))
    p50 = np.exp(models[0.5].predict(Xvb))
    p90 = np.exp(models[0.9].predict(Xvb))
    dva["p50"], dva["p90"] = p50, p90

    MET = {}
    MET["baseline"] = (lambda b: metrics(yv, b[0.5], b[0.9]))(baseline_quantiles(dtr, dva))
    # Ridge (mean-loss) pada log target
    Xr_tr, yr, (Xr_va,) = prepare(dtr, [dva])
    rr = ridge_fit_predict(Xr_tr, yr, dtr.layanan.reset_index(drop=True), [Xr_va], [dva.layanan])[0]
    MET["ridge_log"] = metrics(yv, np.exp(rr[0.5]), np.exp(rr[0.9]))
    MET["lgbm_kuantil_log"] = metrics(yv, p50, p90)
    R["metrik_foldB"] = MET
    bl_ = baseline_quantiles(dtr, dva)
    cm = yv <= CLEAN_THR  # perbandingan pada validasi tanpa outlier ekstrem (sama basis dengan validasi acak)
    R["metrik_foldB_bersih"] = {
        "baseline": metrics(yv[cm], bl_[0.5][cm], bl_[0.9][cm]),
        "ridge_log": metrics(yv[cm], np.exp(rr[0.5])[cm], np.exp(rr[0.9])[cm]),
        "lgbm_kuantil_log": metrics(yv[cm], p50[cm], p90[cm])}
    R["n_val_B"], R["n_val_B_outlier"] = int(len(yv)), int((~cm).sum())
    for k, v in MET.items():
        print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)

# ---- Validasi acak (perbandingan) -------------------------------------------
with Timer("validasi acak"):
    d_ok = tr[tr.ok]
    idx = np.random.default_rng(SEED).permutation(len(d_ok))
    cut = int(0.8 * len(d_ok))
    drt, drv = d_ok.iloc[idx[:cut]], d_ok.iloc[idx[cut:]]
    Xrt, yrt, (Xrv,) = prepare(drt, [drv])
    ma = fit_lgb(Xrt, yrt, 0.5, BEST, n_est=int(IT50 * 1.1))
    mb = fit_lgb(Xrt, yrt, 0.9, BEST, n_est=int(IT90 * 1.1))
    R["validasi_acak"] = metrics(drv.durasi_jam.values, np.exp(ma.predict(Xrv)), np.exp(mb.predict(Xrv)))
    print("acak", R["validasi_acak"], flush=True)

# ---- Dampak kebocoran jumlah_scan dan f30 -----------------------------------
with Timer("dampak leakage"):
    mtr, mva = folds["B"]
    dtr, dva2 = tr[mtr & tr.ok], tr[mva]
    X1, y1, (X1v, X1t) = prepare(dtr, [dva2, te], use_scan=True)
    s5 = fit_lgb(X1, y1, 0.5, BEST, n_est=int(IT50 * 1.1))
    s9 = fit_lgb(X1, y1, 0.9, BEST, n_est=int(IT90 * 1.1))
    R["dengan_scan_foldB"] = metrics(dva2.durasi_jam.values, np.exp(s5.predict(X1v)), np.exp(s9.predict(X1v)))
    cm2 = dva2.durasi_jam.values <= CLEAN_THR
    R["dengan_scan_foldB_bersih"] = metrics(dva2.durasi_jam.values[cm2], np.exp(s5.predict(X1v))[cm2],
                                            np.exp(s9.predict(X1v))[cm2])
    # efek pada test: median prediksi P50
    Xt_clean = prepare(dtr, [te])[2][0]
    R["scan_pred_median_test"] = float(np.median(np.exp(s5.predict(X1t))))
    R["bersih_pred_median_test"] = float(np.median(np.exp(models[0.5].predict(Xt_clean))))
    R["median_durasi_train"] = float(tr.durasi_jam.median())
    X2, y2, (X2v,) = prepare(dtr, [dva2], use_f30=True)
    f5 = fit_lgb(X2, y2, 0.5, BEST, n_est=int(IT50 * 1.1))
    R["dengan_f30_pin50"] = pinball(dva2.durasi_jam.values, np.exp(f5.predict(X2v)), 0.5)
    R["dengan_f30_pin50_bersih"] = pinball(dva2.durasi_jam.values[cm2], np.exp(f5.predict(X2v))[cm2], 0.5)
    print({k: R[k] for k in ["dengan_scan_foldB", "scan_pred_median_test", "bersih_pred_median_test",
                              "dengan_f30_pin50"]}, flush=True)

# ---- Dampak outlier (model loss rata-rata: Ridge) ---------------------------
with Timer("dampak outlier"):
    mtr, mva = folds["B"]
    dtr_all, dva3 = tr[mtr], tr[mva]
    Xo_, yo_, (Xov,) = prepare(dtr_all, [dva3])  # termasuk outlier
    ro = ridge_fit_predict(Xo_, yo_, dtr_all.layanan.reset_index(drop=True), [Xov], [dva3.layanan])[0]
    R["ridge_dengan_outlier"] = metrics(dva3.durasi_jam.values, np.exp(ro[0.5]), np.exp(ro[0.9]))
    R["ridge_tanpa_outlier"] = MET["ridge_log"]

# ---- Kategori baru: simulasi kurir tak dikenal -------------------------------
with Timer("kategori baru"):
    held = "K12"
    mtr, mva = folds["B"]
    dtr_h = tr[mtr & tr.ok & (tr.mitra_kurir != held)]
    dva_h = tr[mva & (tr.mitra_kurir == held)]
    res = {}
    for name, p in [("tanpa_masking", 0.0), ("dengan_masking", 0.04)]:
        Xh, yh, (Xhv,) = prepare(dtr_h, [dva_h], mask_p=p)
        mh = fit_lgb(Xh, yh, 0.5, BEST, n_est=int(IT50 * 1.1))
        res[name] = pinball(dva_h.durasi_jam.values, np.exp(mh.predict(Xhv)), 0.5)
    res["n_val"] = int(len(dva_h))
    R["simulasi_kurir_baru"] = res
    print(res, flush=True)

# ----------------------------------------------------------------------------
# 5. KALIBRASI CONFORMAL (Mondrian per layanan, dalam skala log)
# ----------------------------------------------------------------------------
def conformal_offsets(y, p90, groups, alpha=0.1):
    """Offset aditif (skala log) per kelompok sehingga cakupan P90 ~ 90%."""
    s = np.log(y) - np.log(p90)
    out = {}
    for g in np.unique(groups):
        sg = s[groups == g]
        n = len(sg)
        k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        out[g] = float(np.quantile(sg, k))
    return out


with Timer("conformal"):
    dva = dva.sort_values("tgl")
    cal = dva[dva.tgl < "2025-08-01"]  # kalibrasi: Juli
    chk = dva[dva.tgl >= "2025-08-01"]  # uji cakupan: Agustus
    off = conformal_offsets(cal.durasi_jam.values, cal.p90.values, cal.layanan.values)
    adj = chk.p90.values * np.exp(chk.layanan.map(off).values)
    R["conformal_offset_juli"] = off
    R["conformal_agustus"] = dict(
        cov_sebelum=float(np.mean(chk.durasi_jam.values <= chk.p90.values)),
        cov_sesudah=float(np.mean(chk.durasi_jam.values <= adj)),
        pin90_sebelum=pinball(chk.durasi_jam.values, chk.p90.values, 0.9),
        pin90_sesudah=pinball(chk.durasi_jam.values, adj, 0.9))
    print(R["conformal_agustus"], flush=True)
    # offset final dari seluruh validasi fold B (Jul-Agu)
    OFF_FINAL = conformal_offsets(dva.durasi_jam.values, dva.p90.values, dva.layanan.values)
    R["conformal_offset_final"] = OFF_FINAL
    dva["p90c"] = dva.p90 * np.exp(dva.layanan.map(OFF_FINAL).values)
    dva["p90c"] = np.maximum(dva.p90c, dva.p50 * 1.001)
    R["metrik_foldB_kalibrasi_penuh"] = metrics(dva.durasi_jam.values, dva.p50.values, dva.p90c.values)

# ----------------------------------------------------------------------------
# 6. ANALISIS ERROR + SHAP
# ----------------------------------------------------------------------------
with Timer("analisis"):
    def breakdown(col):
        rows = {}
        for k, g in dva.groupby(col):
            rows[str(k)] = dict(n=int(len(g)), pin50=pinball(g.durasi_jam, g.p50, 0.5),
                                pin90=pinball(g.durasi_jam, g.p90, 0.9),
                                cov90=float(np.mean(g.durasi_jam <= g.p90)),
                                cov90_kal=float(np.mean(g.durasi_jam <= g.p90c)))
        return rows

    R["error_layanan"] = breakdown("layanan")
    R["error_pulau"] = breakdown("pulau_tujuan")
    R["error_mitra"] = breakdown("mitra_kurir")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    pp = pd.DataFrame(R["error_pulau"]).T.sort_values("pin50")
    ax[0].barh(pp.index, pp.pin50, color="#3b6fb6")
    ax[0].set_title("Pinball q=0.5 per pulau tujuan (validasi B)")
    ax[0].set_xlabel("jam")
    pp = pd.DataFrame(R["error_pulau"]).T.loc[pp.index]
    ax[1].barh(pp.index, pp.cov90, color="#d9822b", label="sebelum")
    ax[1].barh(pp.index, pp.cov90_kal, color="none", edgecolor="k", label="setelah kalibrasi")
    ax[1].axvline(0.9, color="r", ls="--")
    ax[1].set_xlim(0.5, 1.0)
    ax[1].set_title("Cakupan P90 per pulau")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(HASIL / "fig_regresi_error.png", dpi=150)
    plt.close()

    import shap

    samp = Xvb.sample(3000, random_state=SEED)
    expl = shap.TreeExplainer(models[0.5])
    sv = expl.shap_values(samp)
    imp = pd.Series(np.abs(sv).mean(0), index=samp.columns).sort_values(ascending=False)
    R["shap_top"] = imp.head(15).round(4).to_dict()
    top = imp.head(12).index[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].barh(top, imp[top], color="#3b6fb6")
    ax[0].set_title("SHAP: rata-rata |nilai| (model P50, skala log)")
    j = list(samp.columns).index("jarak_km")
    ax[1].scatter(samp.jarak_km, sv[:, j], s=3, alpha=0.4, c="#d9822b")
    ax[1].set_xlabel("jarak_km")
    ax[1].set_ylabel("kontribusi SHAP (log jam)")
    ax[1].set_title("Dependensi SHAP: jarak_km")
    plt.tight_layout()
    plt.savefig(HASIL / "fig_regresi_shap.png", dpi=150)
    plt.close()

    # figur kebocoran jumlah_scan dan drift f30
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(tr.jumlah_scan.clip(upper=40), bins=40, alpha=0.6, density=True, label="train")
    ax[0].hist(te.jumlah_scan.clip(upper=40), bins=40, alpha=0.6, density=True, label="test")
    ax[0].set_title("jumlah_scan: distribusi train vs test")
    ax[0].legend()
    mm = pd.Series(A["f30_per_bulan"])
    ax[1].plot(mm.index, mm.values, marker="o")
    ax[1].set_title("Rata-rata f30 per bulan (tren waktu)")
    ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(HASIL / "fig_regresi_audit.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------------
# 7. MODEL FINAL + PREDIKSI TEST
# ----------------------------------------------------------------------------
with Timer("final"):
    d_all = tr[tr.ok]
    Xf, yf, (Xt,) = prepare(d_all, [te], mask_p=0.04)
    fin = {}
    for q, it in [(0.5, IT50), (0.9, IT90)]:
        fin[q] = fit_lgb(Xf, yf, q, BEST, n_est=int(it * 1.2))
    t50 = np.exp(fin[0.5].predict(Xt))
    t90 = np.exp(fin[0.9].predict(Xt)) * np.exp(te.layanan.map(OFF_FINAL).values)
    t90 = np.maximum(t90, t50 * 1.001)
    sub = pd.DataFrame({"id_paket": te.id_paket, "p50": t50.round(3), "p90": t90.round(3)})
    assert len(sub) == len(te) and sub.id_paket.is_unique and (sub.p90 >= sub.p50).all()
    sub.to_csv(ROOT / "prediksi_regresi.csv", index=False)
    R["test_pred"] = dict(p50_median=float(np.median(t50)), p90_median=float(np.median(t90)),
                          train_median=float(tr.durasi_jam.median()))
    print(R["test_pred"], flush=True)

save_json(R, "regresi.json")
print("SELESAI regresi")
