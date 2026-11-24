"""Tugas 3 - Clustering: segmentasi pelanggan.

Alur:
  1. Preprocessing: log untuk variabel miring, CLR untuk data komposisional, imputasi, penskalaan, winsorizing.
  2. Pemilihan fitur dengan bukti (eta^2 antar-cluster awal + struktur PCA).
  3. Tiga metode (KMeans, Gaussian Mixture, Ward aglomeratif) dibandingkan.
  4. Pemilihan jumlah cluster dengan silhouette, BIC (GMM), Calinski-Harabasz dan stabilitas (ARI bootstrap).
  5. Profil segmen + anomali (label -1).
Keluaran: cluster.csv, hasil/clustering.json, hasil/fig_cluster_*.png
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from common import DATA, HASIL, ROOT, SEED, Timer, save_json, set_seed

warnings.filterwarnings("ignore")
set_seed()
rng = np.random.default_rng(SEED)
R = {}

df_all = pd.read_csv(DATA / "clustering.csv.gz")
R["n"] = len(df_all)
# Pelanggan tanpa pesanan 90 hari (frekuensi 0, 5,6 persen) adalah segmen sendiri yang trivial: semua metrik belanja
# nol. Jika ikut diklaster, struktur ini mendominasi jarak dan menyembunyikan struktur perilaku pelanggan aktif
# (terbukti: eta^2 = 1 pada percobaan pertama). Mereka dijadikan satu segmen terpisah; klaster hanya pada pelanggan aktif.
inactive = (df_all.frekuensi_90h == 0).values
df = df_all[~inactive].reset_index(drop=True)
R["n_aktif"], R["n_tanpa_pesanan"] = int(len(df)), int(inactive.sum())

# ----------------------------------------------------------------------------
# 1. PREPROCESSING
# ----------------------------------------------------------------------------
P_KAT = [c for c in df.columns if c.startswith("porsi_") and "jam" not in c]
P_JAM = [c for c in df.columns if c.startswith("porsi_jam")]
N_COLS = [f"n{i:02d}" for i in range(1, 16)]
LOG_COLS = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan", "retur_90h",
            "ulasan_90h", "sesi_per_minggu", "durasi_sesi_menit", "lama_member_bulan", "wishlist",
            "klik_iklan_30h", "belanja_30h", "frekuensi_30h"]
PCT_COLS = ["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod"]
OTHER = ["rating_beri_rata2", "rasio_konversi", "umur", "kota_tier", "jumlah_alamat"]

A = {}
A["rata2_keranjang_missing_pct"] = float(df_all.rata2_keranjang.isna().mean() * 100)
A["frekuensi_nol_pct"] = float((df_all.frekuensi_90h == 0).mean() * 100)
A["missing_sama_frekuensi_nol"] = bool((df_all.rata2_keranjang.isna() == (df_all.frekuensi_90h == 0)).all())
A["rasio_konversi_gt1_pct"] = float((df.rasio_konversi > 1).mean() * 100)
A["jumlah_kolom_n"] = len(N_COLS)
A["recency_max"], A["recency_p99"] = float(df.recency_hari.max()), float(df.recency_hari.quantile(0.99))
A["komposisi_kat_sum_max_dev"] = float((df[P_KAT].sum(axis=1) - 1).abs().max())
A["belanja30_gt_belanja90_pct"] = float((df.belanja_30h > df.belanja_90h + 1).mean() * 100)
A["frek30_gt_frek90_pct"] = float((df.frekuensi_30h > df.frekuensi_90h).mean() * 100)
A["perangkat"] = df.perangkat.value_counts().to_dict()
R["audit"] = A


def clr(M, eps=0.005):
    M = np.asarray(M, dtype=float) + eps
    M = M / M.sum(axis=1, keepdims=True)
    L = np.log(M)
    return L - L.mean(axis=1, keepdims=True)


def build_features(d):
    X = pd.DataFrame(index=d.index)
    tak_belanja = d.frekuensi_90h == 0
    for c in LOG_COLS:
        v = d[c].fillna(0)  # rata2_keranjang kosong hanya bila tidak ada pesanan -> 0
        X["log_" + c] = np.log1p(v)
    kat = clr(d[P_KAT])
    for i, c in enumerate(P_KAT):
        X["clr_" + c] = kat[:, i]
    jam = clr(d[P_JAM])
    for i, c in enumerate(P_JAM):
        X["clr_" + c] = jam[:, i]
    for c in PCT_COLS:
        p = d[c].clip(0.001, 0.999)
        X["logit_" + c] = np.log(p / (1 - p))
    X["rating_beri_rata2"] = d.rating_beri_rata2
    X["rasio_konversi"] = d.rasio_konversi.clip(upper=1.0)
    X["umur"] = d.umur
    X["kota_tier"] = d.kota_tier
    X["jumlah_alamat"] = d.jumlah_alamat
    for p_ in ["android", "ios", "web"]:
        X["perangkat_" + p_] = (d.perangkat == p_).astype(float)
    # kolom CRM anonim: log untuk yang miring, apa adanya untuk yang lain
    for c in N_COLS:
        v = d[c]
        if v.min() >= 0 and v.skew() > 2:
            X[c] = np.log1p(v)
        elif v.min() < 0 and v.abs().max() > 100:
            X[c] = np.sign(v) * np.log1p(v.abs())
        else:
            X[c] = v
    return X


Xall = build_features(df)
Z = (Xall - Xall.mean()) / Xall.std()
Z = Z.clip(-5, 5)  # winsorizing pada 5 simpangan baku (outlier ekstrem tidak mendominasi jarak)
R["n_fitur_awal"] = Z.shape[1]

# ----------------------------------------------------------------------------
# 2. SELEKSI FITUR
# ----------------------------------------------------------------------------
with Timer("seleksi fitur"):
    def eta2_by_labels(Zm, labels):
        tot = Zm.var(axis=0, ddof=0)
        within = np.zeros(Zm.shape[1])
        for k in np.unique(labels):
            m = labels == k
            within += m.mean() * Zm[m].var(axis=0, ddof=0)
        return 1 - within / tot

    sub = rng.choice(len(Z), 40000, replace=False)
    eta_runs = []
    for k in (4, 6, 8):
        km = KMeans(k, n_init=2, random_state=SEED).fit(Z.values[sub])
        eta_runs.append(eta2_by_labels(Z.values[sub], km.labels_))
    eta_all = pd.Series(np.mean(eta_runs, axis=0), index=Z.columns).sort_values(ascending=False)
    R["eta2_awal"] = eta_all.round(4).to_dict()

    # PCA: komponen dengan nilai eigen ~1 yang tidak memuat struktur = noise; periksa korelasi kolom n dengan non-n
    nonn = [c for c in Z.columns if c not in N_COLS]
    corr = Z.iloc[sub].corr(method="spearman")
    maxcorr_n = {c: float(corr.loc[c, nonn].abs().max()) for c in N_COLS}
    R["n_maxcorr_dengan_fitur_lain"] = {k: round(v, 3) for k, v in maxcorr_n.items()}
    # kandidat noise: eta^2 rendah pada semua K
    THR = 0.02
    drop = [c for c in Z.columns if eta_all[c] < THR]
    R["fitur_dibuang"] = drop
    keep = [c for c in Z.columns if c not in drop]
    R["fitur_dipakai"] = keep
    print("dibuang:", drop, flush=True)
    print("dipakai:", len(keep), flush=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    e = eta_all.sort_values()
    ax.barh(e.index, e.values, color=["#c0504d" if v < THR else "#3b6fb6" for v in e.values])
    ax.axvline(THR, color="k", ls="--", lw=0.8)
    ax.set_title(r"Pemilihan fitur: $\eta^2$ antar-cluster awal (rata-rata K=4,6,8)")
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.savefig(HASIL / "fig_cluster_fitur.png", dpi=150)
    plt.close()

Zs = Z[keep].values
pca = PCA(n_components=min(15, Zs.shape[1]), random_state=SEED).fit(Zs)
ev = np.cumsum(pca.explained_variance_ratio_)
R["pca_var_kumulatif"] = ev.round(3).tolist()
NPC = int(np.searchsorted(ev, 0.85) + 1)
NPC = min(max(NPC, 6), 15)
R["n_pc"] = NPC
Zp = pca.transform(Zs)[:, :NPC]

# ----------------------------------------------------------------------------
# 3-4. METODE + JUMLAH CLUSTER
# ----------------------------------------------------------------------------
sil_idx = rng.choice(len(Zs), 15000, replace=False)
KS = list(range(2, 13))
sel = []
with Timer("jumlah cluster"):
    for k in KS:
        km = KMeans(k, n_init=3, random_state=SEED).fit(Zs)
        gm = GaussianMixture(k, covariance_type="full", n_init=1, random_state=SEED, max_iter=150).fit(Zp)
        r = dict(k=k, inertia=float(km.inertia_),
                 sil_kmeans=float(silhouette_score(Zs[sil_idx], km.labels_[sil_idx])),
                 ch_kmeans=float(calinski_harabasz_score(Zs, km.labels_)),
                 db_kmeans=float(davies_bouldin_score(Zs, km.labels_)),
                 bic_gmm=float(gm.bic(Zp)), aic_gmm=float(gm.aic(Zp)))
        sel.append(r)
        print(r, flush=True)
    R["seleksi_k"] = sel

    # gap statistic (referensi: data seragam pada kotak PCA), subsampel 15k
    Zg = Zp[sil_idx]
    lo, hi = Zg.min(0), Zg.max(0)
    gap = []
    for k in KS:
        w = np.log(KMeans(k, n_init=2, random_state=SEED).fit(Zg).inertia_)
        wr = [np.log(KMeans(k, n_init=1, random_state=b).fit(rng.uniform(lo, hi, Zg.shape)).inertia_)
              for b in range(4)]
        gap.append(dict(k=k, gap=float(np.mean(wr) - w), s=float(np.std(wr) * np.sqrt(1 + 1 / 4))))
    R["gap"] = gap

# stabilitas bootstrap: dua subsampel 70% -> label pada irisan -> ARI
with Timer("stabilitas"):
    stab = {}
    for k in range(3, 11):
        aris = []
        for b in range(6):
            r1 = np.random.default_rng(b * 2)
            r2 = np.random.default_rng(b * 2 + 1)
            i1 = r1.choice(len(Zs), int(0.7 * len(Zs)), replace=False)
            i2 = r2.choice(len(Zs), int(0.7 * len(Zs)), replace=False)
            k1 = KMeans(k, n_init=1, random_state=b).fit(Zs[i1])
            k2 = KMeans(k, n_init=1, random_state=b + 100).fit(Zs[i2])
            aris.append(adjusted_rand_score(k1.predict(Zs[sil_idx]), k2.predict(Zs[sil_idx])))
        stab[k] = dict(mean=float(np.mean(aris)), sd=float(np.std(aris)))
        print("stab", k, stab[k], flush=True)
    R["stabilitas_kmeans"] = stab

# ----------------------------------------------------------------------------
# Pilih K (konsensus kriteria) -- ditetapkan di bawah setelah data diperiksa
# ----------------------------------------------------------------------------
import json

CHOSEN = json.loads(open(ROOT / "kode" / "k_terpilih.json").read())["k"] if (ROOT / "kode" / "k_terpilih.json").exists() else None
if CHOSEN is None:
    # fallback otomatis: K dengan silhouette tertinggi di antara K yang stabilitasnya (ARI) >= 0.8
    cand = [r for r in sel if r["k"] in stab and stab[r["k"]]["mean"] >= 0.8]
    CHOSEN = max(cand, key=lambda r: r["sil_kmeans"])["k"] if cand else sel[int(np.argmax([r["sil_kmeans"] for r in sel]))]["k"]
R["k_terpilih"] = int(CHOSEN)
K = int(CHOSEN)
print("K terpilih:", K, flush=True)

with Timer("tiga metode"):
    km = KMeans(K, n_init=10, random_state=SEED).fit(Zs)
    gm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=SEED, max_iter=300).fit(Zp)
    lab_km, lab_gm = km.labels_, gm.predict(Zp)
    # Ward aglomeratif pada subsampel lalu perluas dengan kNN
    sw = rng.choice(len(Zp), 12000, replace=False)
    ag = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Zp[sw])
    knn = KNeighborsClassifier(15).fit(Zp[sw], ag.labels_)
    lab_ag = knn.predict(Zp)
    meth = {"kmeans": lab_km, "gmm": lab_gm, "ward": lab_ag}
    R["perbandingan_metode"] = {
        m: dict(silhouette=float(silhouette_score(Zs[sil_idx], l[sil_idx])),
                davies_bouldin=float(davies_bouldin_score(Zs, l)),
                calinski_harabasz=float(calinski_harabasz_score(Zs, l)),
                ukuran=np.bincount(l, minlength=K).tolist()) for m, l in meth.items()}
    R["ari_antar_metode"] = {f"{a}-{b}": float(adjusted_rand_score(meth[a], meth[b]))
                             for a, b in [("kmeans", "gmm"), ("kmeans", "ward"), ("gmm", "ward")]}
    # stabilitas tiap metode pada K terpilih (bootstrap)
    st = {}
    for name in ["kmeans", "gmm"]:
        aris = []
        for b in range(5):
            i1 = np.random.default_rng(b * 2).choice(len(Zs), int(0.7 * len(Zs)), replace=False)
            i2 = np.random.default_rng(b * 2 + 1).choice(len(Zs), int(0.7 * len(Zs)), replace=False)
            if name == "kmeans":
                a_ = KMeans(K, n_init=1, random_state=b).fit(Zs[i1]).predict(Zs[sil_idx])
                b_ = KMeans(K, n_init=1, random_state=b + 100).fit(Zs[i2]).predict(Zs[sil_idx])
            else:
                a_ = GaussianMixture(K, covariance_type="full", random_state=b, max_iter=150).fit(Zp[i1]).predict(Zp[sil_idx])
                b_ = GaussianMixture(K, covariance_type="full", random_state=b + 100, max_iter=150).fit(Zp[i2]).predict(Zp[sil_idx])
            aris.append(adjusted_rand_score(a_, b_))
        st[name] = dict(mean=float(np.mean(aris)), sd=float(np.std(aris)))
    R["stabilitas_metode_K"] = st
    print(R["perbandingan_metode"], R["ari_antar_metode"], st, flush=True)

# metode final: dipilih berdasarkan stabilitas + silhouette (ditulis ke R, dapat diubah lewat k_terpilih.json)
final_name = json.loads(open(ROOT / "kode" / "k_terpilih.json").read()).get("metode", "kmeans") if (ROOT / "kode" / "k_terpilih.json").exists() else "kmeans"
labels = meth[final_name].copy()
R["metode_final"] = final_name

# ----------------------------------------------------------------------------
# 5. ANOMALI (-1)
# ----------------------------------------------------------------------------
with Timer("anomali"):
    iso = IsolationForest(n_estimators=200, max_samples=20000, random_state=SEED, n_jobs=1).fit(Zp)
    s_iso = -iso.score_samples(Zp)  # makin besar makin anomali
    ll = gm.score_samples(Zp)
    # jarak ke pusat cluster sendiri (skala simpangan baku jarak dalam cluster)
    d_c = np.linalg.norm(Zs - km.cluster_centers_[km.labels_], axis=1)
    zc = np.zeros(len(d_c))
    for k in range(K):
        m = km.labels_ == k
        med, mad = np.median(d_c[m]), np.median(np.abs(d_c[m] - np.median(d_c[m]))) * 1.4826
        zc[m] = (d_c[m] - med) / mad
    R["anomali_dist_z_kuantil"] = {str(q): float(np.quantile(zc, q)) for q in (0.5, 0.9, 0.99, 0.995, 0.999, 1.0)}
    R["anomali_iso_kuantil"] = {str(q): float(np.quantile(s_iso, q)) for q in (0.5, 0.9, 0.99, 0.995, 0.999, 1.0)}
    R["anomali_ll_kuantil"] = {str(q): float(np.quantile(ll, q)) for q in (0.0, 0.001, 0.005, 0.01, 0.05, 0.5)}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].hist(np.clip(zc, None, np.quantile(zc, 0.9999)), bins=120, color="#3b6fb6")
    ax[0].set_yscale("log")
    ax[0].set_title("z-robust jarak ke pusat cluster")
    ax[1].hist(s_iso, bins=120, color="#d9822b")
    ax[1].set_yscale("log")
    ax[1].set_title("skor Isolation Forest")
    plt.tight_layout()
    plt.savefig(HASIL / "fig_cluster_anomali.png", dpi=150)
    plt.close()
    np.save(HASIL / "_zc.npy", zc)
    np.save(HASIL / "_iso.npy", s_iso)

# Histogram z-robust menunjukkan dua mode: populasi normal dan "bukit" kedua jauh dari pusat (anomali).
# Ambang = lembah histogram di antara kedua mode (ditentukan otomatis), bukan persentase tetap.
hist, edges = np.histogram(zc, bins=np.linspace(0, np.quantile(zc, 0.9999), 60))
cent = (edges[:-1] + edges[1:]) / 2
hs = np.convolve(hist, np.ones(3) / 3, mode="same")
rng_mask = (cent > 2.5) & (cent < 6.0)
valley = float(cent[rng_mask][np.argmin(hs[rng_mask])])
is_anom = zc >= valley
R["anomali_lembah_z"] = valley
R["n_anomali"] = int(is_anom.sum())
R["anomali_persen"] = float(is_anom.mean() * 100)
R["anomali_iso_p95_persen"] = float((s_iso[is_anom] > np.quantile(s_iso, 0.95)).mean() * 100)
R["anomali_median_fitur"] = {c: float(df.loc[is_anom, c].median()) for c in ["recency_hari", "frekuensi_90h", "belanja_90h", "rasio_konversi"]}
labels_final = labels.copy()
labels_final[is_anom] = -1
SEG_PASIF = K  # nomor cluster untuk pelanggan tanpa pesanan
labels_full = np.full(len(df_all), SEG_PASIF, dtype=int)
labels_full[~inactive] = labels_final

# ----------------------------------------------------------------------------
# PROFIL SEGMEN
# ----------------------------------------------------------------------------
prof_cols = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan",
             "persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod", "retur_90h",
             "sesi_per_minggu", "rasio_konversi", "lama_member_bulan", "umur", "kota_tier", "wishlist",
             "klik_iklan_30h", "porsi_jam_pagi", "porsi_jam_malam"] + P_KAT
prof = df_all.assign(cluster=labels_full).groupby("cluster")[prof_cols].median()
prof["ukuran"] = pd.Series(labels_full).value_counts().sort_index()
prof["persen"] = (prof["ukuran"] / len(df_all) * 100).round(2)
R["profil_median"] = prof.round(3).to_dict(orient="index")
# rata-rata (untuk variabel yang median-nya nol)
R["profil_mean"] = df_all.assign(cluster=labels_full).groupby("cluster")[prof_cols].mean().round(3).to_dict(orient="index")
glob = df_all[prof_cols].median()
ratio = (prof[prof_cols] / glob.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
R["profil_rasio_ke_global"] = ratio.round(2).to_dict(orient="index")

fig, ax = plt.subplots(figsize=(11, 5))
show = [c for c in prof_cols if c not in ("kota_tier",)]
H = (df_all.assign(cluster=labels_full).groupby("cluster")[show].mean()
     - df_all[show].mean()) / df_all[show].std()
im = ax.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_xticks(range(len(show)))
ax.set_xticklabels(show, rotation=75, fontsize=6)
ax.set_yticks(range(len(H)))
ax.set_yticklabels([f"{i} (n={int((labels_full == i).sum())})" for i in H.index], fontsize=7)
plt.colorbar(im, label="z-skor rata-rata cluster")
ax.set_title("Profil segmen: selisih rata-rata cluster terhadap populasi (z)")
plt.tight_layout()
plt.savefig(HASIL / "fig_cluster_profil.png", dpi=150)
plt.close()

# visual PCA 2D + pilihan K
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ks = [r["k"] for r in sel]
ax[0].plot(ks, [r["sil_kmeans"] for r in sel], marker="o")
ax[0].axvline(K, color="r", ls="--", lw=0.8)
ax[0].set_title("Silhouette (KMeans)")
ax[0].set_xlabel("K")
ax[1].plot(ks, [r["bic_gmm"] for r in sel], marker="o", color="g")
ax[1].axvline(K, color="r", ls="--", lw=0.8)
ax[1].set_title("BIC (GMM)")
ax[1].set_xlabel("K")
kk = sorted(stab)
ax[2].errorbar(kk, [stab[k]["mean"] for k in kk], yerr=[stab[k]["sd"] for k in kk], marker="o", color="purple")
ax[2].axvline(K, color="r", ls="--", lw=0.8)
ax[2].set_title("Stabilitas: ARI bootstrap")
ax[2].set_xlabel("K")
plt.tight_layout()
plt.savefig(HASIL / "fig_cluster_k.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
si = rng.choice(len(Zp), 20000, replace=False)
ax.scatter(Zp[si, 0], Zp[si, 1], c=labels_final[si], s=2, cmap="tab10")
ax.set_title("Cluster pada dua komponen PCA pertama (anomali = -1)")
plt.tight_layout()
plt.savefig(HASIL / "fig_cluster_pca.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# KELUARAN
# ----------------------------------------------------------------------------
out = pd.DataFrame({"id_pelanggan": df_all.id_pelanggan, "cluster": labels_full.astype(int)})
assert len(out) == len(df_all) and out.id_pelanggan.is_unique
out.to_csv(ROOT / "cluster.csv", index=False)
R["ukuran_final"] = pd.Series(labels_full).value_counts().sort_index().to_dict()
save_json(R, "clustering.json")
print("SELESAI clustering", R["ukuran_final"])
