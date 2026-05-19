"""
=============================================================
  PROGRAM ENHANCEMENT CITRA RADIOGRAFI MENGGUNAKAN CLAHE
  Versi Final v3 — Semua Perbaikan Terintegrasi
=============================================================
Library  : OpenCV, NumPy, Matplotlib
Fitur    : - Baca TIFF 16-bit dari hasil konversi DICOM
           - Enhancement dengan CLAHE
           - Kalibrasi skala piksel per mm
           - Seleksi ROI lingkaran 10mm
           - Perhitungan SNR & CNR
           - Histogram berbasis data 16-bit asli
           - Tampilan ROI tanpa banding
           - Simpan TIFF + PNG + Grafik per sesi
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# VARIABEL GLOBAL
# ============================================================

kalibrasi_points = []


# ============================================================
# FUNGSI HELPER TAMPILAN
# ============================================================

def resize_untuk_tampilan(image, lebar_max=900):
    """Resize untuk tampilan layar saja — data tidak berubah."""
    h, w = image.shape[:2]
    if w > lebar_max:
        skala = lebar_max / w
        kecil = cv2.resize(image, None, fx=skala, fy=skala,
                           interpolation=cv2.INTER_AREA)
        return kecil, skala
    return image.copy(), 1.0


def normalisasi_8bit_global(image):
    """
    Normalisasi gambar 16-bit ke 8-bit menggunakan
    windowing persentil 2-98 untuk kontras optimal.
    """
    arr = image.astype(np.float32)
    p2  = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 > p2:
        arr = np.clip(arr, p2, p98)
        arr = (arr - p2) / (p98 - p2) * 255
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def normalisasi_8bit_lokal(image, y, x, h, w):
    """
    Crop ROI dan normalisasi LOKAL menggunakan persentil 1-99.
    Menghilangkan banding pada tampilan ROI.
    """
    roi = image[y:y+h, x:x+w].astype(np.float32)
    p1  = np.percentile(roi, 1)
    p99 = np.percentile(roi, 99)
    if p99 > p1:
        roi = np.clip(roi, p1, p99)
        roi = (roi - p1) / (p99 - p1) * 255
    else:
        roi = np.zeros_like(roi)
    return roi.astype(np.uint8)


def safe_val(v):
    """Ganti nilai inf/nan dengan 0 untuk keperluan grafik."""
    if v is None:
        return 0.0
    try:
        if np.isinf(v) or np.isnan(v):
            return 0.0
    except Exception:
        pass
    return float(v)


# ============================================================
# FUNGSI UTAMA
# ============================================================

def load_image(path):
    """Membaca gambar TIFF 16-bit atau format lainnya."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(
            f"Gambar tidak ditemukan: '{path}'\n"
            f"Pastikan nama file dan lokasinya sudah benar."
        )
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        image = image.astype(np.uint16) * 257
    elif image.dtype != np.uint16:
        image = image.astype(np.uint16)

    print(f"[OK] Gambar dimuat   : {path}")
    print(f"     Ukuran          : {image.shape[1]} x {image.shape[0]} piksel")
    print(f"     Bit depth       : {image.dtype}")
    print(f"     Nilai min       : {image.min()}")
    print(f"     Nilai max       : {image.max()}")
    print(f"     Nilai mean      : {image.mean():.0f}")
    return image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Menerapkan CLAHE pada gambar 16-bit."""
    clahe    = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    print(f"[OK] CLAHE diterapkan | clipLimit={clip_limit} | "
          f"tileGridSize={tile_grid_size}")
    return enhanced


def hitung_snr(roi_signal, roi_noise):
    """SNR = mean(signal) / std(noise)"""
    mean_signal = np.mean(roi_signal)
    std_noise   = np.std(roi_noise)
    if std_noise == 0:
        return float('inf')
    return mean_signal / std_noise


def hitung_cnr(roi_signal, roi_background):
    """CNR = |mean(signal) - mean(background)| / std(background)"""
    mean_signal = np.mean(roi_signal)
    mean_bg     = np.mean(roi_background)
    std_bg      = np.std(roi_background)
    if std_bg == 0:
        return float('inf')
    return abs(mean_signal - mean_bg) / std_bg


# ============================================================
# FUNGSI KALIBRASI SKALA
# ============================================================

def klik_kalibrasi(event, x, y, flags, param):
    global kalibrasi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        kalibrasi_points.append((x, y))
        print(f"  [TITIK {len(kalibrasi_points)}] x={x}, y={y}")


def kalibrasi_skala(image):
    """Kalibrasi skala piksel per mm dengan tampilan layar wajar."""
    global kalibrasi_points
    kalibrasi_points = []

    print("\n" + "=" * 60)
    print("  KALIBRASI SKALA")
    print("=" * 60)
    print("  Klik 2 titik pada gambar yang jarak aslinya diketahui.")
    print("  Tekan ENTER untuk konfirmasi. Tekan R untuk reset.\n")

    jarak_asli_mm = float(
        input("  Masukkan jarak asli antara 2 titik (mm): ")
    )

    img_kecil, skala_tampil = resize_untuk_tampilan(
        normalisasi_8bit_global(image), lebar_max=900
    )
    img_tampil = cv2.cvtColor(img_kecil, cv2.COLOR_GRAY2BGR)

    nama_window = "KALIBRASI -- Klik 2 titik lalu ENTER"
    cv2.namedWindow(nama_window)
    cv2.setMouseCallback(nama_window, klik_kalibrasi)

    while True:
        tampil = img_tampil.copy()
        for i, pt in enumerate(kalibrasi_points):
            cv2.circle(tampil, pt, 5, (0, 255, 0), -1)
            cv2.putText(tampil, f"P{i+1}", (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(kalibrasi_points) == 2:
            cv2.line(tampil, kalibrasi_points[0],
                     kalibrasi_points[1], (0, 255, 255), 2)
            jarak_px      = np.sqrt(
                (kalibrasi_points[1][0]-kalibrasi_points[0][0])**2 +
                (kalibrasi_points[1][1]-kalibrasi_points[0][1])**2
            )
            jarak_px_asli = jarak_px / skala_tampil
            px_per_mm     = jarak_px_asli / jarak_asli_mm
            teks = (f"{jarak_px_asli:.1f}px = {jarak_asli_mm}mm | "
                    f"Skala: {px_per_mm:.2f}px/mm")
            cv2.putText(tampil, teks, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        cv2.imshow(nama_window, tampil)
        key = cv2.waitKey(1)
        if key == 13 and len(kalibrasi_points) == 2:
            break
        elif key == ord('r'):
            kalibrasi_points = []
            print("  [RESET] Titik kalibrasi dihapus, klik ulang.")

    cv2.destroyAllWindows()

    jarak_px      = np.sqrt(
        (kalibrasi_points[1][0]-kalibrasi_points[0][0])**2 +
        (kalibrasi_points[1][1]-kalibrasi_points[0][1])**2
    )
    jarak_px_asli = jarak_px / skala_tampil
    piksel_per_mm = jarak_px_asli / jarak_asli_mm

    print(f"\n  [OK] Jarak piksel   : {jarak_px_asli:.2f} px")
    print(f"  [OK] Jarak asli     : {jarak_asli_mm} mm")
    print(f"  [OK] Skala          : {piksel_per_mm:.4f} piksel/mm")
    return piksel_per_mm


# ============================================================
# FUNGSI SELEKSI ROI LINGKARAN
# ============================================================

def select_roi_lingkaran(image, piksel_per_mm, target_mm=10.0,
                          window_title="Klik tengah ROI"):
    """
    Pilih ROI lingkaran dengan tampilan layar yang wajar.
    Koordinat otomatis dikonversi ke ukuran gambar asli.
    """
    global kalibrasi_points
    kalibrasi_points = []

    radius_piksel = int((target_mm / 2) * piksel_per_mm)

    print(f"\n  [INFO] Target diameter    : {target_mm} mm")
    print(f"  [INFO] Radius dalam piksel: {radius_piksel} px")
    print(f"  Klik titik TENGAH area ROI.")
    print(f"  Tekan ENTER untuk konfirmasi, R untuk mengulang.\n")

    img_kecil, skala_tampil = resize_untuk_tampilan(
        normalisasi_8bit_global(image), lebar_max=900
    )
    img_tampil    = cv2.cvtColor(img_kecil, cv2.COLOR_GRAY2BGR)
    radius_tampil = max(3, int(radius_piksel * skala_tampil))

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, klik_kalibrasi)

    center_tampil = None
    while True:
        tampil = img_tampil.copy()
        if len(kalibrasi_points) >= 1:
            center_tampil = kalibrasi_points[-1]
            cv2.circle(tampil, center_tampil,
                       radius_tampil, (0, 255, 0), 2)
            cv2.circle(tampil, center_tampil, 4,
                       (0, 255, 0), -1)
            info = (f"Diameter: {target_mm}mm ({radius_piksel*2}px)"
                    f" | ENTER=OK  R=ulang")
            cv2.putText(tampil, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)

        cv2.imshow(window_title, tampil)
        key = cv2.waitKey(1)
        if key == 13 and center_tampil is not None:
            break
        elif key == ord('r'):
            kalibrasi_points = []
            center_tampil    = None
            print("  [RESET] Klik ulang titik tengah ROI.")

    cv2.destroyAllWindows()

    # Konversi koordinat tampilan → koordinat gambar asli
    cx_asli     = int(center_tampil[0] / skala_tampil)
    cy_asli     = int(center_tampil[1] / skala_tampil)
    center_asli = (cx_asli, cy_asli)

    x = max(0, cx_asli - radius_piksel)
    y = max(0, cy_asli - radius_piksel)
    w = radius_piksel * 2
    h = radius_piksel * 2

    # Pastikan bounding box tidak keluar dari gambar
    img_h, img_w = image.shape[:2]
    x = min(x, img_w - w)
    y = min(y, img_h - h)

    # Buat mask lingkaran pada gambar ASLI
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(mask, center_asli, radius_piksel, 255, -1)

    print(f"  [OK] Titik tengah ROI : {center_asli}")
    print(f"  [OK] Diameter aktual  : "
          f"{radius_piksel*2}px = {target_mm}mm")
    return (x, y, w, h), center_asli, radius_piksel, mask


# ============================================================
# FUNGSI VISUALISASI
# ============================================================

def plot_hasil(original, enhanced,
               roi_orig_signal, roi_enh_signal,
               roi_orig_bg,     roi_enh_bg,
               roi_signal_coords, center_s, radius_s,
               roi_bg_coords,     center_b, radius_b,
               snr_before, snr_after,
               cnr_before, cnr_after,
               path_output_grafik):
    """Tampilkan dan simpan semua hasil visualisasi."""

    # Normalisasi global untuk tampilan gambar penuh
    orig_8 = normalisasi_8bit_global(original)
    enh_8  = normalisasi_8bit_global(enhanced)

    # Handle nilai inf/nan untuk grafik
    snr_b = safe_val(snr_before)
    snr_a = safe_val(snr_after)
    cnr_b = safe_val(cnr_before)
    cnr_a = safe_val(cnr_after)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.45, wspace=0.35)

    title_color   = '#e0e0e0'
    accent_color  = '#00c8ff'
    accent2_color = '#ff6b6b'
    bg_axes       = '#1a1d27'

    def style_ax(ax, title):
        ax.set_facecolor(bg_axes)
        ax.set_title(title, color=title_color, fontsize=10,
                     fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333644')
        ax.tick_params(colors='#888888', labelsize=8)

    # ── Baris 1: Gambar penuh + ROI ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_8, cmap='gray', vmin=0, vmax=255)
    ax1.add_patch(plt.Circle(center_s, radius_s,
                  color='#00ff88', fill=False,
                  linewidth=2, label='ROI Signal'))
    ax1.add_patch(plt.Circle(center_b, radius_b,
                  color='#ff6b6b', fill=False,
                  linewidth=2, label='ROI Background'))
    ax1.legend(loc='upper right', fontsize=6,
               facecolor='#1a1d27', labelcolor='white',
               edgecolor='#333644')
    style_ax(ax1, "Citra Asli + ROI")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enh_8, cmap='gray', vmin=0, vmax=255)
    ax2.add_patch(plt.Circle(center_s, radius_s,
                  color='#00ff88', fill=False, linewidth=2))
    ax2.add_patch(plt.Circle(center_b, radius_b,
                  color='#ff6b6b', fill=False, linewidth=2))
    style_ax(ax2, "Citra CLAHE + ROI")
    ax2.axis('off')

    # ── ROI Signal dengan normalisasi LOKAL (tanpa banding) ──
    sx, sy, sw, sh = roi_signal_coords

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(normalisasi_8bit_lokal(original, sy, sx, sh, sw),
               cmap='gray', vmin=0, vmax=255)
    style_ax(ax3, "ROI Signal — Sebelum")
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(normalisasi_8bit_lokal(enhanced, sy, sx, sh, sw),
               cmap='gray', vmin=0, vmax=255)
    style_ax(ax4, "ROI Signal — Sesudah CLAHE")
    ax4.axis('off')

    # ── Baris 2: Histogram berbasis data 16-bit ──
    ax5 = fig.add_subplot(gs[1, 0:2])

    # Histogram dari data 16-bit dengan 512 bin
    hist_orig, bins = np.histogram(
        original.flatten(), bins=512,
        range=(int(original.min()), int(original.max()))
    )
    hist_enh, _    = np.histogram(
        enhanced.flatten(), bins=512,
        range=(int(original.min()), int(original.max()))
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax5.fill_between(bin_centers, hist_orig, alpha=0.5,
                     color=accent_color,  label='Asli')
    ax5.fill_between(bin_centers, hist_enh,  alpha=0.5,
                     color=accent2_color, label='CLAHE')
    ax5.set_xlabel('Nilai Piksel (16-bit)',
                   color='#888888', fontsize=8)
    ax5.set_ylabel('Frekuensi', color='#888888', fontsize=8)
    ax5.legend(facecolor='#1a1d27', labelcolor='white',
               edgecolor='#333644', fontsize=8)
    # Format sumbu x agar lebih mudah dibaca
    ax5.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x):,}')
    )
    style_ax(ax5, "Histogram Perbandingan — Citra Penuh (16-bit)")

    # Histogram ROI Signal berbasis data 16-bit
    ax6 = fig.add_subplot(gs[1, 2:4])
    roi_s_orig = original[mask_s_global == 255].flatten()
    roi_s_enh  = enhanced[mask_s_global == 255].flatten()

    if len(roi_s_orig) > 0:
        rmin = int(min(roi_s_orig.min(), roi_s_enh.min()))
        rmax = int(max(roi_s_orig.max(), roi_s_enh.max()))
        if rmax > rmin:
            h_ro, b_ro = np.histogram(roi_s_orig, bins=256,
                                       range=(rmin, rmax))
            h_re, _    = np.histogram(roi_s_enh,  bins=256,
                                       range=(rmin, rmax))
            bc = (b_ro[:-1] + b_ro[1:]) / 2
            ax6.fill_between(bc, h_ro, alpha=0.5,
                             color=accent_color,  label='ROI Asli')
            ax6.fill_between(bc, h_re, alpha=0.5,
                             color=accent2_color, label='ROI CLAHE')
            ax6.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{int(x):,}')
            )

    ax6.set_xlabel('Nilai Piksel (16-bit)',
                   color='#888888', fontsize=8)
    ax6.set_ylabel('Frekuensi', color='#888888', fontsize=8)
    ax6.legend(facecolor='#1a1d27', labelcolor='white',
               edgecolor='#333644', fontsize=8)
    style_ax(ax6, "Histogram Perbandingan — ROI Signal (16-bit)")

    # ── Baris 3: Bar chart SNR/CNR + Tabel statistik ──
    ax7 = fig.add_subplot(gs[2, 0:2])
    labels = ['SNR\nSebelum', 'SNR\nSesudah',
              'CNR\nSebelum', 'CNR\nSesudah']
    values = [snr_b, snr_a, cnr_b, cnr_a]
    colors = [accent_color, '#00ff88', accent2_color, '#ffaa00']
    bars   = ax7.bar(labels, values, color=colors,
                     width=0.5, edgecolor='#333644')

    max_val = max(v for v in values if v > 0) if any(
        v > 0 for v in values) else 1
    for bar, val, raw in zip(bars, values,
                              [snr_before, snr_after,
                               cnr_before, cnr_after]):
        label_txt = ('inf' if (raw is not None and
                               isinstance(raw, float) and
                               np.isinf(raw))
                     else f'{val:.2f}')
        ax7.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max_val * 0.01,
                 label_txt, ha='center', va='bottom',
                 color=title_color, fontsize=9,
                 fontweight='bold')
    ax7.set_ylabel('Nilai', color='#888888', fontsize=8)
    style_ax(ax7, "Perbandingan SNR & CNR")

    # Tabel statistik
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')
    style_ax(ax8, "Ringkasan Statistik")

    def fmt_snr(v):
        if v is not None and isinstance(v, float) and np.isinf(v):
            return 'inf (BG homogen)'
        return f'{v:.4f}' if v is not None else '-'

    stats = [
        ["Parameter",            "Sebelum CLAHE",
         "Sesudah CLAHE"],
        ["Mean (Signal)",
         f"{np.mean(roi_orig_signal):.2f}",
         f"{np.mean(roi_enh_signal):.2f}"],
        ["Std Dev (Signal)",
         f"{np.std(roi_orig_signal):.2f}",
         f"{np.std(roi_enh_signal):.2f}"],
        ["Min (Signal)",
         f"{np.min(roi_orig_signal)}",
         f"{np.min(roi_enh_signal)}"],
        ["Max (Signal)",
         f"{np.max(roi_orig_signal)}",
         f"{np.max(roi_enh_signal)}"],
        ["Mean (Background)",
         f"{np.mean(roi_orig_bg):.2f}",
         f"{np.mean(roi_enh_bg):.2f}"],
        ["Std Dev (Background)",
         f"{np.std(roi_orig_bg):.2f}",
         f"{np.std(roi_enh_bg):.2f}"],
        ["SNR",  fmt_snr(snr_before), fmt_snr(snr_after)],
        ["CNR",  fmt_snr(cnr_before), fmt_snr(cnr_after)],
    ]

    cell_colors = []
    for i in range(len(stats)):
        if i == 0:
            cell_colors.append(['#2a3050'] * 3)
        elif i % 2 == 0:
            cell_colors.append(['#1a1d27'] * 3)
        else:
            cell_colors.append(['#20232f'] * 3)

    table = ax8.table(cellText=stats, cellLoc='center',
                      loc='center', cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#333644')
        if row == 0:
            cell.set_text_props(color='#00c8ff',
                                fontweight='bold')
        else:
            cell.set_text_props(color=title_color)

    fig.suptitle("CLAHE Enhancement — Analisis Citra Radiografi",
                 color=accent_color, fontsize=15,
                 fontweight='bold', y=0.98)

    plt.savefig(path_output_grafik, dpi=150,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n[OK] Grafik disimpan : {path_output_grafik}")
    plt.show()
def kalibrasi_dari_microdicom():
    """
    Kalibrasi skala menggunakan koordinat dari MicroDicom.
    Lebih akurat karena menggunakan data koordinat fisik
    yang sudah dihitung oleh software DICOM.

    Cara mendapatkan data dari MicroDicom:
    1. Buka file DICOM di MicroDicom
    2. Hover mouse ke titik pertama → catat X px, Y px
    3. Hover mouse ke titik kedua  → catat X px, Y px
    4. Gunakan tool Measure Distance → catat jarak mm
    5. Masukkan nilai-nilai tersebut di bawah ini
    """
    print("\n" + "=" * 60)
    print("  KALIBRASI DARI MICRODICOM")
    print("=" * 60)
    print("  Masukkan koordinat dari MicroDicom")
    print("  (lihat pojok kanan bawah saat hover di gambar)\n")

    print("  ── Titik 1 ──")
    x1_px = float(input("  X titik 1 (piksel): "))
    y1_px = float(input("  Y titik 1 (piksel): "))

    print("\n  ── Titik 2 ──")
    x2_px = float(input("  X titik 2 (piksel): "))
    y2_px = float(input("  Y titik 2 (piksel): "))

    print("\n  ── Jarak dari Measure Distance ──")
    jarak_mm = float(input("  Jarak (mm) dari MicroDicom: "))

    # Hitung jarak dalam piksel
    jarak_px = np.sqrt((x2_px - x1_px)**2 +
                       (y2_px - y1_px)**2)

    # Hitung skala
    piksel_per_mm = jarak_px / jarak_mm

    print(f"\n  [OK] Titik 1          : ({x1_px}, {y1_px}) px")
    print(f"  [OK] Titik 2          : ({x2_px}, {y2_px}) px")
    print(f"  [OK] Jarak piksel     : {jarak_px:.2f} px")
    print(f"  [OK] Jarak asli       : {jarak_mm} mm")
    print(f"  [OK] Skala            : {piksel_per_mm:.4f} px/mm")
    print(f"  [OK] Resolusi spasial : {1/piksel_per_mm:.4f} mm/px")
    return piksel_per_mm


def kalibrasi_dari_pixel_spacing(path_dicom):
    """
    Kalibrasi otomatis dari metadata Pixel Spacing DICOM.
    Paling akurat — langsung dari data mesin X-Ray.
    """
    try:
        import pydicom
        ds = pydicom.dcmread(path_dicom)

        if hasattr(ds, 'PixelSpacing'):
            ps = ds.PixelSpacing
            # PixelSpacing = [row spacing, col spacing] dalam mm
            piksel_per_mm = 1.0 / float(ps[0])
            print(f"\n  [OK] Pixel Spacing    : {float(ps[0]):.4f} mm/px")
            print(f"  [OK] Skala            : {piksel_per_mm:.4f} px/mm")
            return piksel_per_mm

        elif hasattr(ds, 'ImagerPixelSpacing'):
            ps = ds.ImagerPixelSpacing
            piksel_per_mm = 1.0 / float(ps[0])
            print(f"\n  [OK] Imager Pixel Spacing: {float(ps[0]):.4f} mm/px")
            print(f"  [OK] Skala               : {piksel_per_mm:.4f} px/mm")
            return piksel_per_mm

        else:
            print("\n  [WARN] Pixel Spacing tidak ada di metadata DICOM")
            print("         Beralih ke mode manual...")
            return kalibrasi_skala(None)

    except ImportError:
        print("  [ERROR] pydicom belum terinstall")
        print("          Jalankan: pip install pydicom")
        return kalibrasi_skala(None)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return kalibrasi_skala(None)

# ============================================================
# PROGRAM UTAMA
# ============================================================

# Variabel global untuk mask — diakses oleh plot_hasil
mask_s_global = None

def main():
    global mask_s_global

    print("=" * 60)
    print("  CLAHE ENHANCEMENT — CITRA RADIOGRAFI")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. LOAD GAMBAR
    # ✏️  Ganti NO_SESI setiap kali proses sesi baru (01-20)
    # ----------------------------------------------------------
    NO_SESI     = '01'
    PATH_GAMBAR = f'output/analisis/sesi_{NO_SESI}_asli.tiff'

    original = load_image(PATH_GAMBAR)

    # ----------------------------------------------------------
    # 2. PARAMETER CLAHE
    # ----------------------------------------------------------
    CLIP_LIMIT     = 2.0
    TILE_GRID_SIZE = (8, 8)

    enhanced = apply_clahe(original,
                           clip_limit=CLIP_LIMIT,
                           tile_grid_size=TILE_GRID_SIZE)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # 2.5 KALIBRASI SKALA
    # ----------------------------------------------------------
    # Pilih mode kalibrasi:
    # 'manual'   = klik 2 titik di jendela gambar
    # 'dicom'    = input langsung dari MicroDicom
    # 'otomatis' = hitung dari pixel spacing DICOM
    MODE_KALIBRASI = 'dicom'

    if MODE_KALIBRASI == 'manual':
        piksel_per_mm = kalibrasi_skala(original)

    elif MODE_KALIBRASI == 'dicom':
        piksel_per_mm = kalibrasi_dari_microdicom()

    elif MODE_KALIBRASI == 'otomatis':
        piksel_per_mm = kalibrasi_dari_pixel_spacing(
            f'dicom/sesi_{NO_SESI}_asli.dcm'
        )
        # ----------------------------------------------------------
        # 3. PILIH ROI SIGNAL — Corpus Vertebra L3/L4
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("  LANGKAH 1: Pilih ROI SIGNAL")
        print("  → Klik tepat di tengah corpus vertebra L3 atau L4")
        print("    (tulang terang berbentuk kotak di tengah gambar)")
        print("=" * 60)
        roi_signal, center_s, radius_s, mask_s = select_roi_lingkaran(
            original, piksel_per_mm, target_mm=10.0,
            window_title="ROI SIGNAL — Corpus Vertebra L3/L4"
        )
        mask_s_global = mask_s

    # ----------------------------------------------------------
    # 4. PILIH ROI BACKGROUND — Otot Psoas
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LANGKAH 2: Pilih ROI BACKGROUND")
    print("  → Klik di otot psoas KIRI atau KANAN vertebra")
    print("    (area abu-abu homogen di sisi tulang belakang)")
    print("  → JANGAN klik di area hitam di luar tubuh!")
    print("=" * 60)
    roi_bg, center_b, radius_b, mask_b = select_roi_lingkaran(
        original, piksel_per_mm, target_mm=10.0,
        window_title="ROI BACKGROUND — Otot Psoas"
    )

    # ----------------------------------------------------------
    # 5. AMBIL PIKSEL DALAM MASK LINGKARAN (data 16-bit asli)
    # ----------------------------------------------------------
    roi_orig_signal = original[mask_s == 255]
    roi_enh_signal  = enhanced[mask_s == 255]
    roi_orig_bg     = original[mask_b == 255]
    roi_enh_bg      = enhanced[mask_b == 255]

    # ----------------------------------------------------------
    # 6. HITUNG SNR & CNR
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  HASIL PERHITUNGAN")
    print("=" * 60)

    snr_before = hitung_snr(roi_orig_signal, roi_orig_bg)
    snr_after  = hitung_snr(roi_enh_signal,  roi_enh_bg)
    cnr_before = hitung_cnr(roi_orig_signal, roi_orig_bg)
    cnr_after  = hitung_cnr(roi_enh_signal,  roi_enh_bg)

    def fmt(v):
        return 'inf' if np.isinf(v) else f'{v:.4f}'

    print(f"\n  SNR Sebelum CLAHE : {fmt(snr_before)}")
    print(f"  SNR Sesudah CLAHE : {fmt(snr_after)}")
    if not np.isinf(snr_before) and not np.isinf(snr_after):
        delta = snr_after - snr_before
        print(f"  Perubahan SNR     : "
              f"{'+' if delta >= 0 else ''}{delta:.4f}")

    print(f"\n  CNR Sebelum CLAHE : {fmt(cnr_before)}")
    print(f"  CNR Sesudah CLAHE : {fmt(cnr_after)}")
    if not np.isinf(cnr_before) and not np.isinf(cnr_after):
        delta = cnr_after - cnr_before
        print(f"  Perubahan CNR     : "
              f"{'+' if delta >= 0 else ''}{delta:.4f}")

    print(f"\n  Mean Signal Asli  : {np.mean(roi_orig_signal):.2f}")
    print(f"  Mean Signal CLAHE : {np.mean(roi_enh_signal):.2f}")
    print(f"  Std BG Asli       : {np.std(roi_orig_bg):.2f}")
    print(f"  Std BG CLAHE      : {np.std(roi_enh_bg):.2f}")

    # Peringatan jika SNR/CNR = inf
    if np.isinf(snr_before) or np.isinf(cnr_before):
        print(f"\n  [PERINGATAN] SNR/CNR Sebelum = inf")
        print(f"  Std Dev Background = 0 → ROI Background")
        print(f"  kemungkinan berada di area yang terlalu homogen.")
        print(f"  Coba jalankan ulang dan letakkan ROI Background")
        print(f"  di otot psoas (area abu-abu di sisi vertebra).")

    print("=" * 60)

    # ----------------------------------------------------------
    # 7. SIMPAN GAMBAR HASIL
    # ----------------------------------------------------------
    os.makedirs('output/analisis', exist_ok=True)
    os.makedirs('output/vga',      exist_ok=True)
    os.makedirs('output/grafik',   exist_ok=True)

    # TIFF 16-bit hasil CLAHE
    path_clahe_tiff = f'output/analisis/sesi_{NO_SESI}_clahe.tiff'
    cv2.imwrite(path_clahe_tiff, enhanced)
    print(f"\n[OK] TIFF CLAHE disimpan : {path_clahe_tiff}")

    # PNG 8-bit untuk VGA
    enhanced_8bit  = normalisasi_8bit_global(enhanced)
    path_clahe_png = f'output/vga/sesi_{NO_SESI}_clahe.png'
    cv2.imwrite(path_clahe_png, enhanced_8bit)
    print(f"[OK] PNG VGA disimpan    : {path_clahe_png}")

    # ----------------------------------------------------------
    # 8. TAMPILKAN & SIMPAN GRAFIK
    # ----------------------------------------------------------
    path_grafik = f'output/grafik/sesi_{NO_SESI}_hasil.png'
    plot_hasil(
        original, enhanced,
        roi_orig_signal, roi_enh_signal,
        roi_orig_bg,     roi_enh_bg,
        roi_signal, center_s, radius_s,
        roi_bg,     center_b, radius_b,
        snr_before, snr_after,
        cnr_before, cnr_after,
        path_grafik
    )

    print(f"\n[SELESAI] Sesi {NO_SESI} berhasil diproses!")
    no_berikut = f"{int(NO_SESI)+1:02d}"
    print(f"\n  Untuk sesi berikutnya:")
    print(f"  Ubah NO_SESI = '{NO_SESI}' → '{no_berikut}'")
    print(f"  lalu jalankan ulang program.\n")


# ============================================================
if __name__ == "__main__":
    main()
