# Kontribusi dan Penggunaan Alat AI

> Berkas ini harus disetujui semua anggota. Bagian bertanda **[ISI]** wajib diisi oleh kelompok; Claude tidak mengetahui siapa mengerjakan apa dan tidak mengisinya.

## 1. Peran tiap anggota

| Anggota | Peran / bagian yang dikerjakan | Tautan pull request |
| --- | --- | --- |
| Revalia Putri (101032330225) | Regretion & Clustering | [ISI] |
| Tarannia Raisha Putri (101032300231) | Classification & Clustering | [ISI] |

Ketentuan soal: setiap anggota fork repo kelompok, bekerja di fork sendiri, dan membuka pull request; semua PR harus di-merge dan fork disinkronkan sebelum 24 Nov 2026 23:59.

## 2. Penggunaan alat AI

**Alat:** Claude (Anthropic), lewat antarmuka chat claude.ai, dengan akses ke lingkungan eksekusi Python.

**Bagian yang dibantu AI:**

| Bagian | Bantuan AI |
| --- | --- |
| Audit data (ketiga tugas) | Merancang pemeriksaan (kebocoran fitur, drift, outlier, kematangan label, kategori baru, noise) dan menjalankannya |
| Kode `kode/01-03_*.py` | Menulis dan men-debug seluruh pipeline: fitur, validasi berbasis waktu, model, kalibrasi, analisis error |
| Pemilihan metode | Usulan metode (LightGBM kuantil, conformal Mondrian, aturan biaya p x nominal, CLR + K-Means) |
| `laporan.pdf` | Menyusun teks dan tabel dari keluaran kode (`kode/04_laporan.py`) |
| `README.md`, `kontribusi.md` | Menyusun draf |

**Cara kelompok memverifikasi keluaran AI** (diisi/dikonfirmasi anggota di bagian 3). Verifikasi yang dilakukan selama pembuatan:

- Semua angka di laporan dibaca langsung dari `hasil/*.json` yang ditulis kode; kode dijalankan ulang dan menghasilkan angka yang sama (regresi dan clustering diuji dua kali).
- Format ketiga berkas prediksi dicek terhadap data test: jumlah baris, ID unik dan identik dengan test, p90 >= p50, `prob` di [0,1], `tandai` 0/1, `cluster` bilangan bulat.
- Beberapa klaim awal yang keliru dikoreksi setelah diperiksa ke data (misalnya: pola fraud per kanal ternyata berubah drastis pada Agustus; audit clustering sempat salah menghitung data kosong).
- Gambar SHAP diperiksa visual sebelum dideskripsikan.

**Yang belum diverifikasi independen:** performa pada data test tersembunyi, dan apakah batas kematangan label 10 Agu serta pilihan K=4 adalah yang terbaik. Keduanya adalah keputusan yang perlu kelompok pahami dan bela di ujian oral.

**Catatan asal kode:** `kode/03_clustering.py` dan `kode/k_terpilih.json` mengandung perubahan yang tidak ditulis oleh Claude dalam percakapan ini (pemisahan pelanggan tanpa pesanan, ambang anomali berbasis lembah histogram, K=4 dengan catatan stabilitas). Claude telah membaca dan menjalankan versi tersebut, dan hasilnya konsisten dengan angka yang dihitung ulang. **[ISI]** Anggota yang membuat perubahan itu: ________ (atau tulis "tidak diketahui" jika memang tidak diketahui).

## 3. Persetujuan anggota

Kami menyatakan bahwa kami memahami seluruh pekerjaan kelompok dan siap mereproduksinya pada ujian oral.

- [ ] [Revalia Putri (101032330225)]
- [ ] [Tarannia Raisha Putri (101032300231)]
