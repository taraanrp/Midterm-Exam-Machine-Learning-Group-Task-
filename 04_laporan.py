"""Menyusun laporan.pdf dari hasil/*.json dan hasil/fig_*.png (semua angka berasal dari keluaran skrip 01-03)."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

from common import HASIL, ROOT, load_json

RG, KL, CL = load_json("regresi.json"), load_json("klasifikasi.json"), load_json("clustering.json")

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=12, spaceAfter=4)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=6, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
SM = ParagraphStyle("SM", parent=B, fontSize=7.5, leading=9.5)
story = []


def p(t, st=B):
    story.append(Paragraph(t, st))


def h1(t):
    story.append(Paragraph(t, H1))


def h2(t):
    story.append(Paragraph(t, H2))


def img(name, w=17):
    path = HASIL / name
    iw, ih = PILImage.open(path).size
    story.append(Image(str(path), width=w * cm, height=w * cm * ih / iw))
    story.append(Spacer(1, 4))


def tbl(rows, widths=None, fs=7.5):
    rows = [[Paragraph(str(c), SM) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f3")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t)
    story.append(Spacer(1, 6))


f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
rp = lambda x: f"Rp {x / 1e6:,.1f} jt"
pc = lambda x: f"{x * 100:.1f}%"

# ============================================================================ JUDUL
story.append(Paragraph("Laporan UTS Machine Learning (Tugas Kelompok)", ParagraphStyle("T", parent=H1, fontSize=18)))
p("<b>Hands-on Machine Learning: Regresi, Klasifikasi, dan Clustering</b><br/>"
  "Anggota kelompok: [ISI NAMA DAN NIM SELURUH ANGGOTA]<br/>"
  "Repositori: [ISI TAUTAN GITHUB]. Semua angka dalam laporan ini dihasilkan oleh <font face='Courier'>kode/run_all.sh</font> "
  "(Python 3.12.3, seed 42; versi pustaka di <font face='Courier'>kode/requirements.txt</font>).")

h1("Ringkasan")
mb = RG["metrik_foldB_bersih"]
p(f"<b>Tugas 1 (regresi).</b> LightGBM kuantil pada log durasi, tanpa <i>jumlah_scan</i> dan <i>f30</i>. Pada validasi berbasis waktu "
  f"(latih Jan-Jun, validasi Jul-Agu, tanpa {RG['n_val_B_outlier']} outlier ekstrem), pinball P50 = {f2(mb['lgbm_kuantil_log']['pin50'])} jam "
  f"(baseline grup {f2(mb['baseline']['pin50'])}, Ridge {f2(mb['ridge_log']['pin50'])}). Cakupan P90 sebelum kalibrasi "
  f"{pc(RG['metrik_foldB']['lgbm_kuantil_log']['cov90'])}, setelah conformal per layanan {pc(RG['metrik_foldB_kalibrasi_penuh']['cov90'])}.")
cb = KL["perbandingan_foldB"]
p(f"<b>Tugas 2 (klasifikasi).</b> LightGBM tanpa penanganan imbalance, dilatih hanya pada transaksi berlabel matang (sampai 10 Agu), tanpa "
  f"<i>flag_review_manual</i> (bocor) dan <i>v15</i> (drift). PR-AUC validasi Jul-Agu = {f3(cb['lgbm_biasa']['pr_auc'])} "
  f"(logistik {f3(cb['logistik']['pr_auc'])}). Pola fraud baru di QRIS (merchant muda, mulai 28 Jul) ditangani dengan bobot 5x pada transaksi sejak tanggal itu. Dengan aturan keputusan p x nominal &gt; Rp 25.000, total biaya paruh akhir validasi "
  f"{rp(KL['kebijakan_biaya']['aturan_biaya_platt (p*nominal>25rb)'])} berbanding {rp(KL['kebijakan_biaya']['tidak_tandai_apa_pun'])} (tidak menandai) dan "
  f"{rp(KL['kebijakan_biaya']['tandai_semua'])} (menandai semua).")
p(f"<b>Tugas 3 (clustering).</b> {CL['n_aktif']:,} pelanggan aktif dikelompokkan dengan K-Means pada 36 fitur terpilih menjadi "
  f"{CL['k_terpilih']} segmen; {CL['n_tanpa_pesanan']:,} pelanggan tanpa pesanan menjadi satu segmen tersendiri, dan {CL['n_anomali']:,} pelanggan "
  f"berperilaku ekstrem diberi label -1. Struktur datanya lemah (silhouette sekitar 0,08), sehingga hasil ini harus dibaca dengan hati-hati (lihat Bagian 4).")
p("<b>Catatan jujur tentang metrik.</b> Angka pinball mentah pada validasi (sekitar 11 jam) didominasi sekitar 100 paket dengan durasi ribuan jam. "
  "Kami melaporkan kedua versi (mentah dan tanpa outlier). Skor sebenarnya pada test tersembunyi tidak kami ketahui.")

# ============================================================================ TUGAS 1
story.append(PageBreak())
h1("1. Tugas 1: Regresi, estimasi durasi pengiriman")
A = RG["audit"]
h2("1.a Audit data")
tbl([["Temuan", "Bukti", "Tindakan"],
     ["<i>jumlah_scan</i> tidak tersedia saat pickup (bocor)",
      f"Terkumpul selama pengiriman. Korelasi Spearman dengan durasi {f2(A['corr_scan_durasi'])}. Rata-rata {f2(A['jumlah_scan_mean_train'])} di train vs {f2(A['jumlah_scan_mean_test'])} di test.",
      "Dibuang dari semua model."],
     ["<i>f30</i> bergeser linear terhadap waktu",
      f"Rata-rata per bulan {A['f30_per_bulan']['2025-01']} (Jan) sampai {A['f30_per_bulan']['2025-10']} (Okt). Korelasi dengan durasi {f3(A['corr_f30_durasi'])}.",
      "Dibuang."],
     ["Outlier target",
      f"{A['outlier_gt1000']} paket durasi &gt; 1000 jam (maks {A['durasi_max']:.0f} jam; P99 = {A['durasi_p99']:.0f} jam).",
      "Dibuang dari data latih saja. Validasi tetap dilaporkan dengan dan tanpa outlier."],
     ["<i>jarak_km</i> = 0", f"{A['jarak_nol']} baris, tidak masuk akal.", "Diubah menjadi NaN."],
     ["<i>berat_kg</i> kosong", f"{A['berat_missing_pct']:.1f}% baris.", "Dibiarkan NaN (LightGBM) + penanda kosong."],
     ["Kurir baru di test", f"{', '.join(A['mitra_baru_test'])}: {A['mitra_baru_test_pct']:.1f}% baris test, tidak ada di train.",
      "Kategori tak dikenal -> NaN, dengan masking 4% saat latih."],
     ["<i>antrean_hub_asal</i>, <i>f35</i>, <i>f17</i> sedikit bergeser",
      f"Rata-rata antrean {A['antrean_mean_train']:.0f} (train) vs {A['antrean_mean_test']:.0f} (test).", "Dipertahankan (tren musiman wajar, tidak bocor)."]],
    [4.2 * cm, 8 * cm, 4.8 * cm])
img("fig_regresi_audit.png", 15)
lk = RG
p(f"<b>Dampak kebocoran.</b> Pada validasi Jul-Agu (tanpa outlier) pinball P50 dengan <i>jumlah_scan</i> {f2(lk['dengan_scan_foldB_bersih']['pin50'])} jam "
  f"vs {f2(mb['lgbm_kuantil_log']['pin50'])} tanpanya, sehingga fitur ini tampak sangat membantu. Namun pada test distribusinya runtuh, dan median prediksi P50 "
  f"turun menjadi {f2(lk['scan_pred_median_test'])} jam (tanpa fitur: {f2(lk['bersih_pred_median_test'])}; median train {f2(lk['median_durasi_train'])}). "
  f"Model dengan <i>f30</i> memberi pinball P50 {f2(lk['dengan_f30_pin50_bersih'])} (tanpa outlier), sedikit lebih buruk. "
  f"Dampak outlier pada model log-target (Ridge) kecil: pinball P50 {f3(lk['ridge_dengan_outlier']['pin50'])} (dengan) vs {f3(lk['ridge_tanpa_outlier']['pin50'])} (tanpa); "
  "Pengaruh outlier pada model kuantil tidak kami ukur terpisah.")

h2("1.b Skema validasi")
p("Tes sebenarnya adalah dua bulan ke depan (Sep-Okt), jadi validasi dibuat berbasis waktu: <b>Fold A</b> (latih Jan-Apr, validasi Mei-Jun) hanya untuk "
  "tuning hiperparameter dan jumlah pohon; <b>Fold B</b> (latih Jan-Jun, validasi Jul-Agu) hanya untuk pelaporan dan kalibrasi, sehingga tidak ada tuning yang "
  "menyentuh data pelaporan. Sebagai pembanding, validasi acak 80/20:")
vr = RG["validasi_acak"]
tbl([["Skema", "Pinball P50", "Pinball P90", "Cakupan P90"],
     ["Waktu (Fold B, tanpa outlier)", f2(mb["lgbm_kuantil_log"]["pin50"]), f2(mb["lgbm_kuantil_log"]["pin90"]), pc(mb["lgbm_kuantil_log"]["cov90"])],
     ["Acak 80/20 (tanpa outlier)", f2(vr["pin50"]), f2(vr["pin90"]), pc(vr["cov90"])]],
    [6 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
p(f"Validasi acak lebih optimistis (pinball P50 {f2(vr['pin50'])} vs {f2(mb['lgbm_kuantil_log']['pin50'])}), dan cakupan P90 pada validasi waktu "
  f"({pc(mb['lgbm_kuantil_log']['cov90'])}) di bawah target, menandakan geseran antarwaktu yang tidak terlihat pada validasi acak.")

h2("1.c Model dan tuning")
p("Tiga model: (1) <b>baseline</b> kuantil empiris per layanan x pulau tujuan; (2) <b>Ridge</b> pada log durasi + kuantil residual per layanan; "
  "(3) <b>LightGBM kuantil</b> pada log durasi (kuantil log = log kuantil, sehingga transformasi aman). Fitur kota memakai <i>target encoding</i> out-of-fold "
  "(smoothing 20). Tuning hanya pada Fold A, tiga konfigurasi:")
rows = [["Konfigurasi", "num_leaves", "min_child", "pinball P50", "pinball P90", "iterasi P50/P90"]]
for t in RG["tuning"]:
    rows.append([t["config"], t["num_leaves"], t["min_child_samples"], f2(t["pin50"]), f2(t["pin90"]), f"{t['it50']}/{t['it90']}"])
tbl(rows, [2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 3 * cm, 3 * cm])
bc = RG["best_config"]
p(f"Dipilih konfigurasi dengan rata-rata pinball terkecil (num_leaves={bc['num_leaves']}, min_child_samples={bc['min_child_samples']}, laju belajar 0,1). "
  "Hasil Fold B:")
rm = RG["metrik_foldB"]
rows = [["Model", "P50 (tanpa outlier)", "P90 (tanpa outlier)", "P50 (mentah)", "P90 (mentah)", "Cakupan P90 (mentah)"]]
for k, nm in [("baseline", "Baseline grup"), ("ridge_log", "Ridge (log)"), ("lgbm_kuantil_log", "LightGBM kuantil (log)")]:
    rows.append([nm, f2(mb[k]["pin50"]), f2(mb[k]["pin90"]), f2(rm[k]["pin50"]), f2(rm[k]["pin90"]), pc(rm[k]["cov90"])])
tbl(rows, [4.2 * cm, 2.7 * cm, 2.7 * cm, 2.5 * cm, 2.5 * cm, 2.6 * cm])

h2("1.d Kategori kardinalitas tinggi dan kategori baru")
sk = RG["simulasi_kurir_baru"]
p(f"<i>kota_tujuan</i> ({A['kota_tujuan_nunique']} nilai) dan <i>kota_asal</i> memakai target encoding out-of-fold; kota baru jatuh ke rata-rata global "
  f"(tidak ada kota tujuan baru di test). Kurir <b>K13</b> hanya ada di test, jadi kami mensimulasikannya: kurir K12 dikeluarkan dari data latih Fold B lalu dievaluasi "
  f"pada {sk['n_val']} paket K12. Pinball P50 {f3(sk['dengan_masking'])} dengan masking 4% saat latih vs {f3(sk['tanpa_masking'])} tanpa masking. "
  "Selisihnya kecil, tetapi masking mengajarkan model jalur 'kurir tak dikenal', sehingga dipakai.")

h2("1.e Ketidakpastian: P50, P90, dan kalibrasi")
ca = RG["conformal_agustus"]
p("P90 dari model memiliki cakupan di bawah 90% pada periode berikutnya. Kami mengkalibrasi dengan <b>conformal prediction ala Mondrian</b> pada skala log: "
  "offset aditif per layanan diambil dari kuantil skor nonkonformitas pada data kalibrasi. Uji out-of-time: kalibrasi pada Juli, uji pada Agustus.")
tbl([["Agustus (uji)", "Cakupan P90", "Pinball P90"],
     ["Sebelum kalibrasi", pc(ca["cov_sebelum"]), f2(ca["pin90_sebelum"])],
     ["Setelah conformal", pc(ca["cov_sesudah"]), f2(ca["pin90_sesudah"])]], [6 * cm, 4 * cm, 4 * cm])
off = RG["conformal_offset_final"]
p("Offset final dari seluruh validasi Jul-Agu (skala log): " + ", ".join(f"{k} {v:+.3f}" for k, v in off.items()) +
  f". Dengan offset ini cakupan P90 pada validasi menjadi {pc(RG['metrik_foldB_kalibrasi_penuh']['cov90'])}. Kami memaksa p90 &#8805; p50 pada keluaran.")

h2("1.f Analisis error, SHAP")
img("fig_regresi_error.png", 16)
el = RG["error_layanan"]
rows = [["Layanan", "n", "pinball P50", "pinball P90", "cakupan P90", "cakupan setelah kalibrasi"]]
for k, v in el.items():
    rows.append([k, v["n"], f2(v["pin50"]), f2(v["pin90"]), pc(v["cov90"]), pc(v["cov90_kal"])])
tbl(rows, [3 * cm, 2 * cm, 3 * cm, 3 * cm, 3 * cm, 3.6 * cm])
em = RG["error_mitra"]
worst = max(em.items(), key=lambda kv: kv[1]["pin50"])
best_m = min(em.items(), key=lambda kv: kv[1]["pin50"])
ep = RG["error_pulau"]
wp = max(ep.items(), key=lambda kv: kv[1]["pin50"])
p(f"Error terbesar ada pada layanan <b>hemat</b> (pinball P50 {f2(el['hemat']['pin50'])}) dan paling kecil pada <b>ekspres</b> ({f2(el['ekspres']['pin50'])}); "
  f"pulau tujuan dengan error terbesar adalah <b>{wp[0]}</b> ({f2(wp[1]['pin50'])}). Kurir {worst[0]} paling sulit ({f2(worst[1]['pin50'])}), "
  f"{best_m[0]} paling mudah ({f2(best_m[1]['pin50'])}). Setelah kalibrasi cakupan P90 tiap layanan berada di sekitar 90%, tetapi cakupan per pulau "
  "belum dijamin karena kalibrasi hanya per layanan.")
img("fig_regresi_shap.png", 16)
sh = RG["shap_top"]
top = list(sh.items())[:6]
p("SHAP (model P50, skala log, 3.000 sampel validasi). Fitur terpenting: " + ", ".join(f"<i>{k}</i> ({v:.3f})" for k, v in top) +
  ". Jenis layanan dan jarak mendominasi, diikuti efek kota tujuan (target encoding) dan beberapa fitur anonim (f17, f04). Kontribusi jarak naik "
  "tajam sampai sekitar 2.000 km lalu melandai (panel kanan); beberapa titik di sekitar 10.000 km sedikit lebih rendah, jumlahnya sedikit sehingga tidak kami tafsirkan. Fitur anonim hanya dapat ditafsirkan sebagai 'penting', bukan secara substantif.")
p(f"Prediksi test: median P50 {f2(RG['test_pred']['p50_median'])} jam dan median P90 {f2(RG['test_pred']['p90_median'])} jam "
  f"(median durasi train {f2(RG['test_pred']['train_median'])} jam), konsisten dengan distribusi latih.")

# ============================================================================ TUGAS 2
story.append(PageBreak())
h1("2. Tugas 2: Klasifikasi, deteksi fraud")
A = KL["audit"]
h2("2.a Audit label dan fitur")
wk = A["fraud_rate_mingguan_akhir"]
tbl([["Temuan", "Bukti", "Tindakan"],
     ["Label belum matang di ujung data latih",
      f"Fraud rate matang (1 Jul-10 Agu) {pc(A['rate_matang_Jul_Agu10'])}, tetapi minggu terakhir (25-31 Agu) hanya {pc(A['rate_1minggu_terakhir'])}. "
      "Fraud dikonfirmasi belakangan, sehingga transaksi baru tampak 'bersih' padahal belum terkonfirmasi.",
      "Latih hanya pada transaksi sampai 10 Agu. Validasi dilakukan dengan jeda 21 hari."],
     ["<i>flag_review_manual</i> bocor",
      f"AUC sendirian {f2(A['flag_review_auc'])}; fraud rate {pc(A['flag_review_rate_fraud'])} bila 1 vs {pc(A['flag_review_rate_nonflag'])} bila 0. "
      f"{A['fraud_dengan_flag_pct']:.0f}% fraud punya flag. Ini keputusan tinjauan setelah transaksi, bukan fakta saat transaksi terjadi. "
      f"Porsi flag berbeda: {pc(A['flag_review_share_train'])} (train) vs {pc(A['flag_review_share_test'])} (test).",
      "Dibuang."],
     ["<i>v15</i> bergeser linear",
      f"Rata-rata per bulan {A['v15_per_bulan']['2025-01']} (Jan) sampai {A['v15_per_bulan']['2025-10']} (Okt), AUC terhadap label {f2(A['v15_auc'])} (tanpa sinyal).",
      "Dibuang."],
     ["<i>menit_sejak_trx_terakhir</i> mengikuti panjang riwayat",
      "Median naik dari " + f"{A['menit_median_per_bulan']['2025-01']:.0f} (Jan) ke {A['menit_median_per_bulan']['2025-10']:.0f} (Okt) karena riwayat data dimulai 1 Jan.",
      "Dipotong pada 7 hari, lalu log."],
     ["<i>hari_libur</i> berbeda",
      f"{pc(A['hari_libur_share_train'])} (train) vs {pc(A['hari_libur_share_test'])} (test), tanpa sinyal.", "Dibuang."]],
    [4 * cm, 9 * cm, 4 * cm])
img("fig_klasifikasi_audit.png", 16)
p(f"Pengguna test yang juga ada di train: {A['pengguna_test_di_train_pct']:.0f}%. Fitur riwayat tambahan dibuat hanya dari masa lalu: hitungan transaksi pengguna/merchant/pasangan "
  "dalam 30 hari terakhir, rasio nominal terhadap rata-rata 30 hari, hitungan merchant 1 dan 24 jam, hitungan pengguna 10 menit. Jendela dipakai (bukan hitungan kumulatif) "
  "agar fitur stasioner. Pada percobaan awal dengan hitungan kumulatif, AUC adversarial sedikit lebih tinggi (0,87 vs 0,84), sehingga diganti jendela. Tidak ada fitur berbasis label (label terlambat).")

h2("2.b Validasi berbasis waktu dengan jeda")
p(f"Skema: Fold A (tuning): {KL['folds']['A']}. Fold B (pelaporan): {KL['folds']['B']}. Jeda 21 hari meniru kenyataan bahwa saat model dilatih, label transaksi "
  f"21 hari terakhir belum lengkap. Model final dilatih sampai 10 Agu, lalu dipakai untuk 1 Sep dan seterusnya (jeda yang sama). Fold B: {KL['fold_B']['n_train']:,} latih "
  f"({KL['fold_B']['fraud_train']} fraud), {KL['fold_B']['n_val']:,} validasi ({KL['fold_B']['fraud_val']} fraud).")

h2("2.c Ketidakseimbangan kelas: perbandingan pendekatan")
rows = [["Pendekatan (Fold B)", "PR-AUC", "ROC-AUC"]]
for k, nm in [("logistik", "Regresi logistik (baseline)"), ("lgbm_biasa", "LightGBM tanpa penanganan"),
              ("lgbm_class_weight", f"LightGBM class weight (scale_pos_weight={KL['spw']:.1f})"),
              ("lgbm_undersampling", "LightGBM undersampling negatif 10% + koreksi probabilitas"),
              ("lgbm_recency", "LightGBM + bobot recency (paruh-hidup 45 hari)")]:
    rows.append([nm, f3(cb[k]["pr_auc"]), f3(cb[k]["roc_auc"])])
tbl(rows, [9 * cm, 3 * cm, 3 * cm])
tn = KL["tuning"]
p(f"Tuning Fold A (PR-AUC): num_leaves 15 -> {f3(tn[0]['pr_auc'])}, num_leaves 63 -> {f3(tn[1]['pr_auc'])}; dipilih num_leaves 15. "
  "Class weight dan undersampling tidak mengalahkan model biasa pada PR-AUC (metrik yang dipakai di penilaian). "
  "Tuning threshold dilakukan di tahap keputusan biaya (2.e), bukan pada skor.")

h2("2.d Pattern shift: lonjakan fraud QRIS di merchant muda")
pw = KL["pr_auc_mingguan_foldB"]
rows = [["Minggu validasi (Fold B)", "n", "fraud", "PR-AUC"]] + [[k, v["n"], v["fraud"], f3(v["pr_auc"])] for k, v in pw.items()]
tbl(rows, [6 * cm, 3 * cm, 3 * cm, 3 * cm])
S = KL["pola_baru_qris"]
mixk = KL["fraud_mix_kanal"]["qris"]
p(f"PR-AUC stabil sekitar 0,73-0,78 sampai akhir Juli, lalu turun ke {f3(list(pw.values())[-1]['pr_auc'])} pada 4-10 Agu. Pemeriksaan komposisi fraud menunjukkan penyebabnya: "
  f"porsi QRIS dalam seluruh fraud sekitar {mixk['2025-07'] * 100:.0f}-{mixk['2025-02'] * 100:.0f}% per bulan (Jan-Jul) melonjak menjadi {mixk['2025-08'] * 100:.0f}% pada Agustus, padahal porsi "
  "QRIS dalam seluruh transaksi stabil sekitar 22%. Jadi ini pola fraud baru, bukan kenaikan volume QRIS.")
qm_ = S["qris_mingguan"]
tbl([["Minggu (QRIS)", "n transaksi", "fraud"]] + [[k, v["n"], v["fraud"]] for k, v in qm_.items()], [6 * cm, 4 * cm, 3 * cm])
p(f"Pola baru (28 Jul-10 Agu): {S['n_fraud_qris_baru']} fraud QRIS di {S['n_merchant_baru']} merchant, dengan umur merchant median {S['umur_merchant_median_baru']:.0f} hari "
  f"(fraud QRIS sebelumnya: {S['umur_merchant_median_lama']:.0f} hari), nominal median Rp {S['nominal_median_baru']:,.0f} (sebelumnya Rp {S['nominal_median_lama']:,.0f}), kategori "
  + ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(S["kategori_baru"].items())[:2]) + f" (sebelumnya didominasi fashion, kecantikan, elektronik). Fraud rate pada merchant berumur &#8804; 30 hari naik dari "
  f"{S['fraud_rate_merchant_muda_30h_sebelum'] * 100:.1f}% menjadi {S['fraud_rate_merchant_muda_30h_sesudah'] * 100:.1f}%. Median fitur risiko pengguna pada fraud baru ini (perangkat baru, login gagal, transaksi 1 jam) bernilai nol, "
  "sehingga model yang dilatih sebelum pola ini tidak punya sinyal kuat untuk menangkapnya; ini konsisten dengan kelemahan kanal QRIS di 2.f. Merchant-merchant ini juga muncul di test "
  f"({S['trx_test_di_merchant_ini']} transaksi).")
ad = KL["adaptasi_pola_baru"]
rows = [["Adaptasi (latih sampai 4 Agu, uji 5-10 Agu)", "PR-AUC semua", "PR-AUC QRIS", "biaya aturan (Rp)"]]
for k, nmm in [("tanpa_bobot", "Tanpa bobot"), ("bobot_x5_sejak_28Jul", "Bobot 5x untuk transaksi sejak 28 Jul"),
               ("recency_paruh_14h", "Recency, paruh-hidup 14 hari"), ("recency_paruh_45h", "Recency, paruh-hidup 45 hari")]:
    rows.append([nmm, f3(ad[k]["pr_auc"]), f3(ad[k]["pr_auc_qris"]), f"{ad[k]['biaya_aturan']:,.0f}"])
tbl(rows, [7.5 * cm, 3 * cm, 3 * cm, 3.5 * cm])
p(f"Hanya {ad['tanpa_bobot']['fraud']} fraud ({ad['tanpa_bobot']['fraud_qris']} QRIS) pada jendela uji ini, sehingga kesimpulannya tentatif dan pemilihan varian pada jendela yang sama bersifat optimistis. "
  f"Pembobotan 5x sejak 28 Jul menaikkan PR-AUC dari {f3(ad['tanpa_bobot']['pr_auc'])} menjadi {f3(ad['bobot_x5_sejak_28Jul']['pr_auc'])} (QRIS: {f3(ad['tanpa_bobot']['pr_auc_qris'])} menjadi "
  f"{f3(ad['bobot_x5_sejak_28Jul']['pr_auc_qris'])}); recency 14 hari serupa. Dipilih <b>bobot 5x</b> untuk model final. Bobot recency 45 hari di Fold B tidak membantu "
  f"(PR-AUC {f3(cb['lgbm_recency']['pr_auc'])} vs {f3(cb['lgbm_biasa']['pr_auc'])}) karena Fold B dilatih sampai 15 Juni dan belum melihat pola baru. "
  f"Pada test, model final memberi probabilitas rata-rata {KL['test_merchant_pola_baru']['mean_prob']:.3f} untuk {KL['test_merchant_pola_baru']['n']} transaksi di merchant tersebut dan menandai "
  f"{KL['test_merchant_pola_baru']['n_flag']} di antaranya; kami tidak bisa memverifikasinya tanpa label test.")
p(f"Validasi adversarial (test vs Jul-Agu, hanya kovariat) memberi AUC {f2(KL['adversarial_auc'])}; fitur pembeda utama: " + ", ".join(f"<i>{k}</i>" for k in list(KL['adversarial_top'])[:3]) +
  ". <i>umur_merchant_hari</i> bertambah seiring waktu secara alami, jadi sebagian geseran ini wajar tetapi tidak sepenuhnya kami hilangkan.")

h2("2.e Kalibrasi dan keputusan berbasis biaya")
kk = KL["kalibrasi"]
tbl([["Paruh akhir validasi B", "Brier", "ECE", "rata-rata prob"],
     ["Mentah", f"{kk['mentah']['brier']:.5f}", f"{kk['mentah']['ece']:.5f}", f"{kk['mentah']['mean_p']:.4f}"],
     ["Platt", f"{kk['platt']['brier']:.5f}", f"{kk['platt']['ece']:.5f}", f"{kk['platt']['mean_p']:.4f}"],
     ["Isotonik", f"{kk['isotonik']['brier']:.5f}", f"{kk['isotonik']['ece']:.5f}", f"{kk['isotonik']['mean_p']:.4f}"],
     ["Laju fraud aktual", "", "", f"{kk['rate_aktual']:.4f}"]], [5 * cm, 3 * cm, 3 * cm, 3.5 * cm])
p("Kalibrasi dipasang pada paruh awal validasi B dan diuji pada paruh akhir (urutan waktu). Selisih ketiganya kecil karena LightGBM sudah cukup terkalibrasi; "
  "kami memakai Platt (dua parameter, monoton sehingga PR-AUC tidak berubah, dan lebih stabil daripada isotonik).")
img("fig_klasifikasi_kalibrasi.png", 16)
kb = KL["kebijakan_biaya"]
ks = KL["kebijakan_ringkas"]
rows = [["Kebijakan (paruh akhir validasi B)", "Total biaya"]] + [[k, f"Rp {v:,.0f}"] for k, v in kb.items()]
tbl(rows, [10 * cm, 5 * cm])
p(f"Biaya = nominal fraud yang lolos + Rp 25.000 x jumlah transaksi ditandai. Aturan optimal untuk probabilitas terkalibrasi: tandai bila <b>p x nominal &gt; 25.000</b> "
  f"(kerugian yang diharapkan terhindar melebihi biaya tinjauan). Pada {ks['n']:,} transaksi ({ks['fraud']} fraud, nominal fraud {rp(ks['nominal_fraud'])}) aturan ini menandai "
  f"{ks['n_flag_aturan']} transaksi dan menangkap {pc(ks['recall_aturan'])} fraud, dengan biaya {rp(kb['aturan_biaya_platt (p*nominal>25rb)'])}. Itu sekitar "
  f"{kb['aturan_biaya_platt (p*nominal>25rb)'] / kb['tidak_tandai_apa_pun'] * 100:.0f}% dari biaya tidak menandai apa pun dan "
  f"{kb['aturan_biaya_platt (p*nominal>25rb)'] / kb['tandai_semua'] * 100:.0f}% dari biaya menandai semuanya. Threshold global pada p (dituning di paruh awal) lebih mahal "
  f"({rp(kb['threshold_global_p (dituning paruh awal)'])}) karena mengabaikan nominal. Selisih aturan Platt, isotonik, dan tanpa kalibrasi kecil dan dalam rentang noise "
  "(hanya 207 fraud di paruh akhir).")

h2("2.f Analisis error")
ek = KL["error_kanal"]
rows = [["Kanal", "n", "fraud", "PR-AUC", "recall (aturan biaya)", "biaya fraud lolos"]]
for k, v in ek.items():
    rows.append([k, v["n"], v["fraud"], f3(v["pr_auc"]), pc(v["recall"]), rp(v["biaya_miss"])])
tbl(rows, [3 * cm, 2.5 * cm, 2 * cm, 2.5 * cm, 3.5 * cm, 3.5 * cm])
mv = KL["miss_vs_caught_top"]
p(f"Dari {KL['n_missed'] + KL['n_caught']} fraud di seluruh validasi B, {KL['n_caught']} tertangkap dan {KL['n_missed']} lolos. Kanal <b>QRIS</b> paling bermasalah: PR-AUC "
  f"{f3(ek['qris']['pr_auc'])}, recall {pc(ek['qris']['recall'])}, dan {rp(ek['qris']['biaya_miss'])} dari kerugian lolos berasal dari QRIS. Kanal web, VA, dan kartu jauh lebih baik. Kelemahan QRIS konsisten dengan pola baru pada 2.d (model Fold B dilatih sebelum pola itu muncul); ini hipotesis yang sesuai dengan datanya, bukan hasil uji terpisah. "
  f"Fraud yang lolos berpola 'tenang': median nominal {KL['median_nominal_missed']:,.0f} vs {KL['median_nominal_caught']:,.0f} untuk yang tertangkap; "
  f"rata-rata <i>nominal_24j</i> {mv['nominal_24j']['missed']:,.0f} vs {mv['nominal_24j']['caught']:,.0f}, <i>gagal_login_24j</i> {mv['gagal_login_24j']['missed']:.2f} vs "
  f"{mv['gagal_login_24j']['caught']:.2f}, <i>ganti_password_7h</i> {mv['ganti_password_7h']['missed']:.2f} vs {mv['ganti_password_7h']['caught']:.2f}. "
  "Artinya model baik pada fraud yang disertai tanda ambil-alih akun (login gagal, ganti kata sandi, aktivitas padat), dan lemah pada fraud tunggal tanpa riwayat mencurigakan. "
  f"Sebanyak {KL['fraud_nominal_<25rb_pct']:.1f}% fraud bernominal di bawah Rp 25.000, yang secara ekonomi tidak layak ditinjau.")
imp = KL["importance_top"]
p("Fitur terpenting (gain, model LightGBM biasa): " + ", ".join(f"<i>{k}</i> ({v * 100:.1f}%)" for k, v in list(imp.items())[:6]) + ".")
p(f"Prediksi test: rata-rata prob terkalibrasi {KL['test']['mean_prob']:.4f}; {KL['test']['n_flag']:,} transaksi ({pc(KL['test']['flag_rate'])}) ditandai.")

# ============================================================================ TUGAS 3
story.append(PageBreak())
h1("3. Tugas 3: Clustering, segmentasi pelanggan")
A = CL["audit"]
h2("3.a Preprocessing")
p(f"Temuan data: <i>rata2_keranjang</i> kosong pada {A['rata2_keranjang_missing_pct']:.1f}% baris, tepat sama dengan pelanggan dengan frekuensi_90h = 0 (kosong berarti tidak ada pesanan). "
  f"Ada {A['rasio_konversi_gt1_pct']:.2f}% <i>rasio_konversi</i> &gt; 1 (dipotong pada 1). <i>recency_hari</i> maksimum {A['recency_max']:.0f} vs P99 {A['recency_p99']:.0f}. "
  f"Kamus data menyebut atribut CRM <i>n01-n10</i>, tetapi file berisi <b>{A['jumlah_kolom_n']} kolom</b> (n01-n15). Perlakuan: log1p untuk variabel miring (belanja, frekuensi, recency, sesi, dll.); "
  "transformasi <b>CLR</b> (centered log-ratio, dengan penambahan 0,005 untuk nol) untuk porsi kategori dan porsi jam karena jumlahnya selalu 1; logit untuk persentase promo; "
  "penskalaan z lalu pemotongan pada +/-5 simpangan baku agar outlier tidak mendominasi jarak. <i>kota_tier</i> diperlakukan ordinal dan <i>perangkat</i> one-hot.")
p(f"<b>Pelanggan tanpa pesanan.</b> {CL['n_tanpa_pesanan']:,} pelanggan ({CL['n_tanpa_pesanan'] / CL['n'] * 100:.1f}%) tidak punya belanja 90 hari. Semua metrik belanja mereka nol, jadi jika ikut diklaster "
  "mereka akan membentuk kelompok sendiri yang mendominasi jarak. Mereka ditetapkan sebagai satu segmen tersendiri (label 4), dan klaster dibentuk pada "
  f"{CL['n_aktif']:,} pelanggan aktif.")

h2("3.b Seleksi fitur dengan bukti")
img("fig_cluster_fitur.png", 11)
nm = CL["n_maxcorr_dengan_fitur_lain"]
p(f"Bukti: pada K-Means awal (K=4,6,8) dihitung <b>eta kuadrat</b> tiap fitur (porsi varians antar-cluster). Ambang 0,02. Seluruh {A['jumlah_kolom_n']} kolom <i>n</i> berada di bawah ambang, dan korelasi "
  f"Spearman maksimum tiap kolom <i>n</i> dengan fitur non-<i>n</i> hanya {min(nm.values()):.3f}-{max(nm.values()):.3f}, yaitu praktis tidak berhubungan dengan perilaku pelanggan. Mereka dibuang. "
  f"Dibuang juga: " + ", ".join(f"<i>{c}</i>" for c in CL["fitur_dibuang"] if not c.startswith("n")) + f". Fitur dipakai: {len(CL['fitur_dipakai'])}. "
  "Fitur yang paling membedakan: log belanja/frekuensi (90 dan 30 hari), persentase voucher, recency, rata-rata keranjang, ulasan, umur, dan rasio konversi. "
  "Keterbatasan: ambang 0,02 bersifat pilihan kami; fitur dengan eta kuadrat rendah bisa saja informatif untuk pembagian yang tidak dipilih K-Means.")

h2("3.c Metode dan jumlah cluster")
sel = CL["seleksi_k"]
img("fig_cluster_k.png", 16)
rows = [["K", "silhouette", "Calinski-Harabasz", "Davies-Bouldin", "BIC (GMM)", "gap", "stabilitas ARI"]]
gp = {g["k"]: g["gap"] for g in CL["gap"]}
st = CL["stabilitas_kmeans"]
for r in sel:
    k = r["k"]
    rows.append([k, f3(r["sil_kmeans"]), f"{r['ch_kmeans']:.0f}", f2(r["db_kmeans"]), f"{r['bic_gmm']:.0f}", f3(gp[k]),
                 f"{st[str(k)]['mean']:.2f}" if str(k) in st else "-"])
tbl(rows, [1.2 * cm, 2.3 * cm, 3 * cm, 2.6 * cm, 2.7 * cm, 2 * cm, 2.8 * cm])
p(f"<b>Kriteria tidak sepakat.</b> Silhouette tertinggi di K=2 (0,104) tetapi pembagian dua kelompok kurang bermakna untuk pemasaran, dan silhouette di K=3-12 hanya 0,06-0,08. "
  "BIC terus turun hingga sekitar K=9 dan statistik gap naik hingga K=11 tanpa puncak jelas, yang menandakan tidak ada struktur diskret yang tegas. "
  f"Stabilitas bootstrap (ARI antar dua subsampel 70%) turun dari 0,89 (K=3) menjadi 0,80 (K=4) dan 0,64 (K=5). Kami memilih <b>K={CL['k_terpilih']}</b>, K terbesar yang masih stabil "
  "(ARI &#8805; 0,8), dengan silhouette yang setara dengan K=6. Ini kompromi, bukan optimum tegas; kombinasi dua kriteria (stabilitas dan silhouette) mendukungnya, sedangkan BIC dan gap menyarankan K lebih besar.")
pm = CL["perbandingan_metode"]
rows = [["Metode (K=4)", "silhouette", "Davies-Bouldin", "Calinski-Harabasz", "ukuran cluster"]]
for k, nmm in [("kmeans", "K-Means"), ("gmm", "Gaussian Mixture (PCA 15 dim)"), ("ward", "Ward aglomeratif (subsampel 12k + kNN)")]:
    rows.append([nmm, f3(pm[k]["silhouette"]), f2(pm[k]["davies_bouldin"]), f"{pm[k]['calinski_harabasz']:.0f}", str(pm[k]["ukuran"])])
tbl(rows, [5.2 * cm, 2.2 * cm, 2.6 * cm, 3 * cm, 4 * cm])
ar = CL["ari_antar_metode"]
sm = CL["stabilitas_metode_K"]
p(f"ARI antar metode rendah: K-Means vs GMM {f2(ar['kmeans-gmm'])}, K-Means vs Ward {f2(ar['kmeans-ward'])}, GMM vs Ward {f2(ar['gmm-ward'])}. GMM membentuk satu cluster sangat besar (73.198, "
  f"hampir sama dengan jumlah pengguna Android, yang kemungkinan dipicu fitur biner perangkat), sehingga kurang bermakna. Stabilitas bootstrap pada K=4: K-Means {sm['kmeans']['mean']:.2f} "
  f"(sd {sm['kmeans']['sd']:.2f}), GMM {sm['gmm']['mean']:.2f} (sd {sm['gmm']['sd']:.2f}). Karena silhouette dan stabilitas K-Means paling baik dan hasilnya paling mudah ditafsirkan, "
  "metode final adalah <b>K-Means</b>. Simpangan baku stabilitas besar (0,15-0,2), jadi perbedaan antar-K kecil tidak boleh dibaca terlalu jauh.")

h2("3.d Anomali (label -1)")
img("fig_cluster_anomali.png", 14)
am = CL["anomali_median_fitur"]
p(f"Histogram jarak-robust ke pusat cluster memperlihatkan dua mode: populasi normal dan 'bukit' kedua jauh dari pusat. Ambang diambil di lembah antar keduanya (z = {CL['anomali_lembah_z']:.2f}), "
  f"bukan persentase tetap, sehingga menandai {CL['n_anomali']:,} pelanggan ({CL['anomali_persen']:.2f}%). {CL['anomali_iso_p95_persen']:.0f}% di antaranya juga berada di 5% skor teratas Isolation Forest. "
  "Profil anomali berbeda tajam: sesi per minggu rata-rata sekitar 226 (segmen lain 4-8), klik iklan 30 hari sekitar 186 (segmen lain sekitar 6), wishlist sekitar 140 (segmen lain sekitar 11), "
  f"dengan rasio konversi median hanya {am['rasio_konversi']:.3f}. Pola ini mirip bot atau akun penjelajah ekstrem yang jarang membeli.")

h2("3.e Profil dan aksi pemasaran per segmen")
img("fig_cluster_profil.png", 16)
PM = CL["profil_mean"]
NAMES = {
    "0": ("Pembeli Bernilai Tinggi (Elektronik dan Gadget)",
          "keranjang rata-rata Rp 710 ribu (tertinggi), belanja 90 hari Rp 6,3 juta, porsi elektronik dan gadget tinggi, pengguna voucher rendah (21%), usia rata-rata 39 tahun.",
          "Program bundling aksesoris, garansi dan tukar tambah, serta cicilan paylater untuk barang mahal."),
    "1": ("Pelanggan Mulai Pasif",
          "recency 58 hari, hanya 2,8 pesanan dan belanja Rp 538 ribu dalam 90 hari, COD tinggi (43%).",
          "Kampanye win-back: voucher personal berbatas waktu dan pengingat produk dari wishlist."),
    "2": ("Pemburu Promo Muda",
          "usia 26 tahun, voucher 73% dan flash sale 62% dari pesanan, belanja malam (40%), fashion dan kecantikan dominan, retur tertinggi (1,31), anggota baru (14 bulan), keranjang kecil.",
          "Notifikasi flash sale malam hari, panduan ukuran untuk menekan retur, dan insentif naik kelas keranjang (min. belanja)."),
    "3": ("Pembelanja Rutin Rumah Tangga",
          "21 pesanan per 90 hari, 5 barang per pesanan, belanja pagi (38%), makanan-minuman, rumah tangga, dan ibu-bayi dominan, konversi tertinggi (0,29).",
          "Langganan dan pesan ulang otomatis, poin loyalitas, serta pengiriman gratis untuk isi ulang kebutuhan harian."),
    "4": ("Belum Pernah Belanja (90 hari)",
          "tanpa pesanan 90 hari terakhir, sesi aplikasi masih ada, usia anggota tertua (28 bulan).",
          "Voucher 'pesanan pertama kembali' dan onboarding rekomendasi kategori populer."),
    "-1": ("Aktivitas Ekstrem (anomali)",
           "sesi sekitar 226 per minggu, klik iklan sekitar 186, wishlist sekitar 140, konversi sangat rendah.",
           "Keluarkan dari target kampanye dan biaya iklan, periksa dugaan bot atau scraping."),
}
sz = CL["ukuran_final"]
rows = [["Label", "Nama segmen", "n (%)", "Ciri utama", "Satu aksi pemasaran"]]
for lab in ["0", "1", "2", "3", "4", "-1"]:
    n_, d_, a_ = NAMES[lab]
    rows.append([lab, n_, f"{sz[lab]:,} ({sz[lab] / CL['n'] * 100:.1f}%)", d_, a_])
tbl(rows, [1 * cm, 3 * cm, 2.2 * cm, 6.3 * cm, 4.6 * cm])
p("Nama dan ciri disusun dari profil rata-rata dan median tiap segmen (tabel profil lengkap ada di <font face='Courier'>hasil/clustering.json</font>). Karena struktur data lemah, "
  "perbedaan antar-segmen terutama dibentuk oleh nilai belanja, frekuensi, recency, dan sensitivitas promo; porsi kategori memberi perbedaan lebih halus.")
img("fig_cluster_pca.png", 8)
p(f"Ukuran akhir: " + ", ".join(f"{k}: {v:,}" for k, v in sz.items()) + ". Keluaran <font face='Courier'>cluster.csv</font> berisi 120.000 baris, satu per pelanggan.")

# ============================================================================ KETERBATASAN
h1("4. Keterbatasan dan catatan penting")
p("<b>Regresi.</b> Skor mentah pada test akan terpengaruh outlier ekstrem yang tidak dapat diprediksi. Kalibrasi conformal hanya per layanan, bukan per pulau atau kurir. "
  "Kurir K13 tidak punya data latih; prediksinya mengandalkan jalur 'tak dikenal'.")
p("<b>Klasifikasi.</b> Batas kematangan label (10 Agu) dipilih dari pola laju mingguan dan bisa kurang tepat. Adaptasi terhadap pola QRIS baru didasarkan pada hanya 47 fraud dan dipilih pada jendela uji kecil (76 fraud); performa pada Sep-Okt tidak terjamin. "
  "Geseran kovariat (AUC adversarial sekitar 0,84) belum hilang sepenuhnya. Validasi keputusan biaya hanya berdasar 207 fraud, sehingga selisih antar kalibrasi tidak signifikan. "
  "Laju fraud test tidak diketahui; kalibrasi mengandaikan laju mirip periode latih matang (sekitar 0,7%).")
p("<b>Clustering.</b> Struktur lemah (silhouette sekitar 0,08), kriteria pemilihan K tidak sepakat, dan K=4 adalah kompromi. Ambang eta kuadrat 0,02 dan ambang lembah anomali adalah pilihan analis. "
  "Penetapan nama dan aksi segmen bersifat interpretatif.")
h1("5. Penggunaan alat AI")
p("Alat AI (Claude, Anthropic) digunakan untuk membantu merancang pipeline, menulis dan men-debug kode, serta menyusun laporan ini. Rincian per bagian, cara verifikasi, "
  "dan peran tiap anggota ada di <font face='Courier'>kontribusi.md</font>. Seluruh angka dalam laporan ini berasal dari keluaran kode yang dapat dijalankan ulang dengan satu perintah.")

doc = SimpleDocTemplate(str(ROOT / "laporan.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Laporan UTS Machine Learning")
doc.build(story)
print("laporan.pdf selesai")
