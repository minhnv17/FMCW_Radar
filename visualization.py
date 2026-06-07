"""
visualization.py — Hiển thị kết quả FMCW Radar

Module này tạo figure hiển thị toàn bộ kết quả phân tích:
    - Doppler Profile
    - Range Profile
    - Range-Doppler Map (2D FFT)
    - Thông số Radar
    - Phổ không gian Azimuth (Polar Plot)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from radar_config import RadarConfig
from signal_generator import Target
from radar_types import DetectedTarget


def plot_results(
    config: RadarConfig,
    targets: list[Target],
    detected_targets: list[DetectedTarget],
    rdm_db: np.ndarray,
    range_bins: np.ndarray,
    doppler_bins: np.ndarray,
    dynamic_range_db: float = 40.0,
):
    """
    Hiển thị kết quả phân tích FMCW Radar.
    """
    # ── Thiết lập figure ────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    fig.suptitle(
        "FMCW Radar Simulation Results",
        fontsize=16,
        fontweight="bold",
        color="#e0e0ff",
        y=0.97,
    )

    # 2 rows, 3 columns
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1.5, 1.3, 1.0],
        hspace=0.40,
        wspace=0.35,
        left=0.05,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )

    # Style chung cho tất cả axes
    axes_style = {
        "facecolor": "#16213e",
    }
    label_color = "#b0b0d0"
    title_color = "#e0e0ff"
    tick_color = "#8888aa"

    # ────────────────────────────────────────────────────────────
    # Panel 1: Doppler Profile (max over range)
    # ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], **axes_style)
    ax1.set_title("Doppler Profile (Max over Range)",
                  color=title_color, fontsize=12)
    doppler_profile_db = np.max(rdm_db, axis=1)
    ax1.plot(doppler_bins, doppler_profile_db, color="#00d2ff", linewidth=1.2)
    ax1.set_xlabel("Doppler Bin", color=label_color, fontsize=10)
    ax1.set_ylabel("Magnitude [dB]", color=label_color, fontsize=10)
    ax1.set_ylim(-dynamic_range_db, 5)
    ax1.tick_params(colors=tick_color, labelsize=8)
    ax1.grid(True, alpha=0.15, color="#4a4a6a")

    # ────────────────────────────────────────────────────────────
    # Panel 2: Range Profile (single chirp FFT)
    # ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], **axes_style)
    ax2.set_title("Range Profile (Max over Doppler)",
                  color=title_color, fontsize=12)
    range_profile_db = np.max(rdm_db, axis=0)
    ax2.plot(range_bins, range_profile_db, color="#ff6b6b", linewidth=1.2)
    ax2.set_xlabel("Range Bin", color=label_color, fontsize=10)
    ax2.set_ylabel("Magnitude [dB]", color=label_color, fontsize=10)
    ax2.set_ylim(-dynamic_range_db, 5)
    ax2.tick_params(colors=tick_color, labelsize=8)
    ax2.grid(True, alpha=0.15, color="#4a4a6a")

    # ────────────────────────────────────────────────────────────
    # Panel 3: Radar Specifications (text)
    # ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[:, 2], **axes_style)
    ax3.set_title("Configuration & Detected Targets",
                  color=title_color, fontsize=12)
    ax3.axis("off")

    specs_text = config.get_specs_text()

    # Ground Truth Targets
    target_lines = ["\n", "GROUND TRUTH", "─" * 32]
    for i, t in enumerate(targets):
        target_lines.append(
            f"T{i+1}: R={t.range_m:.0f}m v={t.velocity:+.1f}m/s Az={t.azimuth_deg:+.1f}°")

    # Detected Targets
    target_lines.extend(["\n", "DETECTED (CFAR + DBF)", "─" * 32])
    for i, dt in enumerate(detected_targets):
        target_lines.append(
            f"D{i+1}: R={dt.range_m:.1f}m v={dt.velocity:+.1f}m/s Az={dt.azimuth_deg:+.1f}°")

    full_text = specs_text + "\n".join(target_lines)
    ax3.text(
        0.05, 0.98, full_text,
        transform=ax3.transAxes, fontsize=10, fontfamily="monospace",
        color="#c0c0e0", verticalalignment="top", linespacing=1.6,
    )

    # ────────────────────────────────────────────────────────────
    # Panel 4: Range-Doppler Map (2D heatmap)
    # ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0], **axes_style)
    ax4.set_title("Range-Doppler Map (CFAR Detections highlighted)",
                  color=title_color, fontsize=12)

    extent = (range_bins[0], range_bins[-1], doppler_bins[0], doppler_bins[-1])
    im = ax4.imshow(
        rdm_db.T, aspect="auto", extent=extent, origin="lower",
        cmap="viridis", vmin=-dynamic_range_db, vmax=0, interpolation="bilinear",
    )
    ax4.set_xlabel("Range Bin", color=label_color, fontsize=10)
    ax4.set_ylabel("Doppler Bin", color=label_color, fontsize=10)
    ax4.tick_params(colors=tick_color, labelsize=8)

    # Đánh dấu các đỉnh CFAR lên RDM (nhóm theo cell để không đè text)
    from collections import defaultdict
    rdm_points = defaultdict(list)
    for dt in detected_targets:
        rdm_points[(dt.r_idx, dt.d_idx)].append(dt.azimuth_deg)

    for (r, d), az_list in rdm_points.items():
        ax4.scatter(r, d, s=100, facecolors='none',
                    edgecolors='white', linewidths=2)
        az_text = ", ".join([f"{a:.1f}°" for a in az_list])
        ax4.text(r + 2, d + 2, f"Az: {az_text}",
                 color='white', fontsize=8, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Magnitude [dB]", color=label_color, fontsize=9)
    cbar.ax.tick_params(colors=tick_color, labelsize=8)

    # ────────────────────────────────────────────────────────────
    # Panel 5: Polar Angle Spectrum
    # ────────────────────────────────────────────────────────────
    # Đồ thị toạ độ cực (Polar Plot) cho góc phương vị Azimuth
    from matplotlib.projections.polar import PolarAxes
    import typing
    ax_polar = typing.cast(PolarAxes, fig.add_subplot(
        gs[1, 1], projection='polar'))
    ax_polar.set_facecolor("#16213e")
    ax_polar.set_title("Azimuth Angle Spectrum",
                       color=title_color, fontsize=12, pad=15)

    # Đặt góc 0 độ ở hướng Bắc (thẳng đứng lên)
    ax_polar.set_theta_zero_location("N")
    # Tăng góc theo chiều kim đồng hồ (phải = dương, trái = âm)
    ax_polar.set_theta_direction(-1)

    # Chỉ hiển thị quạt -90 đến 90 độ (radar trường nhìn phía trước)
    ax_polar.set_thetamin(-90)
    ax_polar.set_thetamax(90)

    colors = ['#ff4757', '#2ed573', '#1e90ff', '#ffa502']

    # Nhóm theo cell để chỉ vẽ phổ 1 lần cho 1 RDM bin
    plotted_spectra = set()
    for i, dt in enumerate(detected_targets):
        cell = (dt.r_idx, dt.d_idx)
        color = colors[i % len(colors)]

        if dt.angles is None or dt.angle_spectrum is None:
            continue
            
        # Góc tính bằng radian
        theta_rad = np.deg2rad(dt.angles)
        spectrum_plot = dt.angle_spectrum - np.min(dt.angle_spectrum)

        # Vẽ phổ nếu cell này chưa được vẽ
        if cell not in plotted_spectra:
            plotted_spectra.add(cell)
            ax_polar.plot(theta_rad, spectrum_plot, color=color,
                          linewidth=2, label=f"RDM Peak ({dt.r_idx}, {dt.d_idx})")
            ax_polar.fill_between(
                theta_rad, 0, spectrum_plot, color=color, alpha=0.1)

        # Đánh dấu đỉnh azimuth
        az_rad = np.deg2rad(dt.azimuth_deg)
        # Phổ tại điểm azimuth (vì spectrum_plot đã dịch max về 0 nên giá trị ở đỉnh luôn bằng độ chênh lệch so với min)
        # Điểm cao nhất của góc chính là max(spectrum_plot)
        max_idx = np.argmin(np.abs(theta_rad - az_rad))
        plot_radius = spectrum_plot[max_idx]

        ax_polar.scatter([az_rad], [plot_radius], color=color,
                         s=80, edgecolors='white', linewidths=1.5, zorder=5)
        # Ghi chú góc
        ax_polar.text(az_rad, plot_radius + 2, f"{dt.azimuth_deg:.1f}°",
                      color='white', fontsize=8, ha='center', va='bottom', fontweight='bold')

    ax_polar.tick_params(colors=tick_color, labelsize=8)
    ax_polar.grid(True, alpha=0.3, color="#4a4a6a")
    if detected_targets:
        ax_polar.legend(loc='upper right', bbox_to_anchor=(
            1.2, 1.1), facecolor="#16213e", edgecolor="#4a4a6a", labelcolor=label_color)

    # Format viền
    for ax in [ax1, ax2, ax4]:
        for spine in ax.spines.values():
            spine.set_color("#2a2a4a")

    try:
        plt.tight_layout()
    except Exception:
        pass

    plt.show()
