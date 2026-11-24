"""Menampilkan gambar hasil (hasil/fig_*.png) di jendela, satu per satu.

Cara pakai (dari folder kode/):
    python tampilkan_gambar.py              # semua gambar
    python tampilkan_gambar.py regresi      # hanya gambar regresi
    python tampilkan_gambar.py klasifikasi  # hanya gambar klasifikasi
    python tampilkan_gambar.py cluster      # hanya gambar clustering
Tutup jendela gambar untuk melihat gambar berikutnya.
"""
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from common import HASIL

kunci = sys.argv[1] if len(sys.argv) > 1 else ""
berkas = sorted(HASIL.glob(f"fig_{kunci}*.png"))
if not berkas:
    sys.exit(f"Tidak ada gambar fig_{kunci}*.png di {HASIL}. Jalankan skrip 01-03 dulu.")

for f in berkas:
    img = mpimg.imread(f)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(11, 11 * h / w))
    fig.canvas.manager.set_window_title(f.name)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    print("Menampilkan:", f.name, "(tutup jendela untuk lanjut)")
    plt.show()
