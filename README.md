# UTS Machine Learning: Regresi, Klasifikasi, Clustering

Isi branch `main` (sesuai ketentuan soal):

| Berkas | Keterangan |
| --- | --- |
| `laporan.pdf` | Laporan Tugas 1-3 (11 halaman) |
| `kode/` | Seluruh kode + README ini (satu perintah mereproduksi semua angka) |
| `prediksi_regresi.csv` | `id_paket,p50,p90` (60.000 baris, p90 >= p50) |
| `prediksi_klasifikasi.csv` | `id_transaksi,prob,tandai` (106.674 baris) |
| `cluster.csv` | `id_pelanggan,cluster` (120.000 baris; -1 = anomali) |
| `kontribusi.md` | Peran tiap anggota dan penggunaan alat AI |

## Cara mereproduksi

1. Python 3.12.3. Pasang pustaka: `pip install -r kode/requirements.txt`
2. Ekstrak `data_uts_ml.zip` ke folder `data/` (file `*.csv.gz` dan `kamus_data.md`; tidak di-commit karena besar). Alternatif: set variabel lingkungan `DATA_DIR`.
3. Jalankan satu perintah: `bash kode/run_all.sh  (atau, dari dalam kode/: bash run_all.sh)`

Skrip: `01_regresi.py`, `02_klasifikasi.py`, `03_clustering.py`, `04_laporan.py` (membangun `laporan.pdf` dari `hasil/*.json` dan `hasil/fig_*.png`).
Semua angka di laporan berasal dari `hasil/*.json` yang ditulis oleh skrip. Seed = 42 di semua skrip.

Versi pustaka: numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, matplotlib 3.11.2, shap 0.54.1, reportlab 5.0.1.

Catatan: dijalankan pada 1 CPU. Waktu total sekitar 20-25 menit (regresi sekitar 4 menit, klasifikasi sekitar 8 menit, clustering sekitar 4 menit).
Hasil LightGBM dapat sedikit berbeda antar mesin/versi; pada mesin yang sama hasilnya identik (clustering diuji dijalankan dua kali dengan hasil sama).

## Keputusan desain utama (ringkas; detail di laporan)

- **Regresi:** `jumlah_scan` (bocor) dan `f30` (drift waktu) dibuang; validasi berbasis waktu (Fold A untuk tuning, Fold B untuk pelaporan); LightGBM kuantil pada log durasi; P90 dikalibrasi conformal (Mondrian per layanan).
- **Klasifikasi:** `flag_review_manual` (bocor), `v15` dan `hari_libur` dibuang; latih hanya pada label matang (sampai 10 Agu); validasi dengan jeda 21 hari; bobot 5x untuk transaksi sejak 28 Jul (pola fraud QRIS baru); kalibrasi Platt; flag bila p x nominal > Rp 25.000.
- **Clustering:** `n01-n15` dibuang berdasarkan eta kuadrat dan korelasi; pelanggan tanpa pesanan menjadi satu segmen; K-Means K=4 pada pelanggan aktif; anomali (-1) lewat lembah histogram jarak-robust.
