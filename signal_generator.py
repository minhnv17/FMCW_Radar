"""
signal_generator.py — Tạo tín hiệu FMCW Radar

Module này mô phỏng quá trình phát (TX) và thu (RX) tín hiệu FMCW Radar,
bao gồm:
  - Tạo beat signal (IF signal) cho nhiều mục tiêu đồng thời
  - Thêm nhiễu Gaussian (AWGN) với SNR có thể điều chỉnh

Lý thuyết:
    ┌────────────────────────────────────────────────────────────┐
    │  TX: x(t) = cos(2π·f_c·t + π·S·t²)                      │
    │                                                            │
    │  RX: y(t) = α · x(t − τ)  với τ = 2R/c                   │
    │       (có Doppler shift nếu mục tiêu di chuyển)           │
    │                                                            │
    │  Beat signal = mix(TX, RX):                                │
    │    f_beat = S · τ = 2·S·R / c         → Range information  │
    │    f_doppler = 2·v·f_c / c            → Velocity info      │
    │                                                            │
    │  Phase của beat signal qua các chirp chứa thông tin vận tốc│
    └────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass

import numpy as np

from radar_config import RadarConfig, SPEED_OF_LIGHT
from ddma_config import DDMAConfig


@dataclass
class Target:
    """
    Mô tả một mục tiêu radar.

    Attributes:
        range_m:       Khoảng cách tới mục tiêu [m]
        velocity:      Vận tốc hướng tâm [m/s]
                       > 0 = đang menjauh (moving away)
                       < 0 = đang đến gần (approaching)
        azimuth_deg:   Góc phương vị (Azimuth) [độ]
        elevation_deg: Góc tà (Elevation) [độ]
        rcs:           Radar Cross Section — tiết diện phản xạ [m²]
                       Ảnh hưởng đến biên độ tín hiệu nhận.
    """
    range_m: float          # [m]
    velocity: float         # [m/s], dương = ra xa, âm = lại gần
    azimuth_deg: float = 0.0   # [độ], ngang
    elevation_deg: float = 0.0  # [độ], dọc
    rcs: float = 1.0        # [m²]


def generate_beat_signal(
    config: RadarConfig,
    targets: list[Target],
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Tạo ma trận beat signal cho toàn bộ frame (tất cả chirps).

    Quá trình:
        1. Với mỗi chirp m (m = 0, 1, ..., N_chirps - 1):
           - Với mỗi target:
             a. Tính beat frequency:  f_b = 2·S·R / c
             b. Tính Doppler phase:   φ_d = 2π · (2·v·f_c/c) · m·T_c
             c. Tạo beat signal mẫu: exp(j·2π·f_b·t + j·φ_d)
             d. Scale theo RCS (biên độ ~ √RCS)
           - Cộng tất cả target lại
        2. Thêm nhiễu AWGN theo SNR yêu cầu.

    Args:
        config:     Cấu hình radar.
        targets:    Danh sách mục tiêu.
        snr_db:     Signal-to-Noise Ratio [dB]. Mặc định 20 dB.

    Returns:
        beat_signal: Ma trận phức [N_chirps × N_samples].
                     Mỗi hàng = 1 chirp, mỗi cột = 1 mẫu ADC.
    """
    c = SPEED_OF_LIGHT
    Nc = config.n_chirps
    Ns = config.n_samples

    # Trục thời gian fast-time: mẫu ADC trong 1 chirp
    #   t = [0, dt, 2·dt, ..., (Ns-1)·dt]
    #   với dt = T_c / Ns
    t_fast = np.arange(Ns) / config.fs  # [s], shape: (Ns,)

    # Khởi tạo ma trận beat signal (phức)
    beat_signal = np.zeros((Nc, Ns), dtype=np.complex128)

    for target in targets:
        R = target.range_m
        v = target.velocity
        rcs = target.rcs

        # Kiểm tra mục tiêu có nằm trong tầm radar không
        if R > config.max_range:
            print(
                f"  [!] Target at {R:.1f}m exceeds max range {config.max_range:.1f}m")
            continue
        if abs(v) > config.max_velocity:
            print(f"  [!] Target velocity {v:.1f}m/s exceeds max velocity "
                  f"+/-{config.max_velocity:.1f}m/s")

        # ── Beat frequency ──────────────────────────────────────
        # Khi TX và RX được trộn, tần số beat tỷ lệ với khoảng cách:
        #   f_b = 2 · S · R / c
        f_beat = 2 * config.slope * R / c

        # ── Doppler frequency ───────────────────────────────────
        # Doppler shift do chuyển động hướng tâm:
        #   f_d = 2 · v · f_c / c
        f_doppler = 2 * v * config.fc / c

        # ── Biên độ phản xạ ─────────────────────────────────────
        # Biên độ tín hiệu nhận ~ √(RCS) / R²  (radar equation đơn giản)
        # Ở đây dùng √RCS cho đơn giản (bỏ qua path loss chi tiết)
        amplitude = np.sqrt(rcs)

        # ── Tạo beat signal cho từng chirp ──────────────────────
        for m in range(Nc):
            # Phase tích lũy qua các chirp (slow-time):
            #   φ_slow = 2π · f_d · m · T_c
            # Đây chính là thông tin vận tốc, được giải mã bởi Doppler FFT
            phase_slow = 2 * np.pi * f_doppler * m * config.t_chirp

            # Beat signal cho chirp thứ m:
            #   s(t) = A · exp(j·2π·f_b·t + j·φ_slow)
            # Thành phần fast-time (f_b·t) chứa thông tin range
            # Thành phần slow-time (φ_slow) chứa thông tin velocity
            beat_signal[m, :] += amplitude * np.exp(
                1j * (2 * np.pi * f_beat * t_fast + phase_slow)
            )

    # ── Thêm nhiễu AWGN ────────────────────────────────────────
    # SNR = 10·log10(P_signal / P_noise)
    # → P_noise = P_signal / 10^(SNR/10)
    signal_power = np.mean(np.abs(beat_signal) ** 2)
    if signal_power > 0:
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns)
        )
        beat_signal += noise

    return beat_signal


def gen_ddma_signal(config: RadarConfig, targets: list[Target], snr_db: float = 20.0) -> np.ndarray:
    """
    Tạo ma trận beat signal cho hệ thống MIMO sử dụng kỹ thuật DDMA
    (Doppler Division Multiple Access).

    Quá trình:
        1. Tạo ma trận mã hóa pha DDMA cho tất cả TX.
        2. Với mỗi RX: tín hiệu thu được là tổng (superposition) của tất cả TX phát đồng thời.
        3. Thêm nhiễu AWGN.

    Args:
        config:     Cấu hình radar.
        targets:    Danh sách mục tiêu.
        snr_db:     Signal-to-Noise Ratio [dB].

    Returns:
        beat_signal: Ma trận phức [n_rx x n_chirps x n_samples].
    """
    c = SPEED_OF_LIGHT
    Nc = config.n_chirps
    Ns = config.n_samples
    # Cấu trúc ăng ten MIMO
    assert config.antenna_array is not None
    n_rx = config.antenna_array.n_rx
    n_tx = config.antenna_array.n_tx
    tx_pos = config.antenna_array.tx_pos
    rx_pos = config.antenna_array.rx_pos

    # 1. Tạo ma trận phase DDMA cho TX: shape (n_tx, Nc)
    ddma_phase = DDMAConfig.get_phase_matrix(n_tx, Nc, mode="standard")

    t_fast = np.arange(Ns) / config.fs
    beat_signal = np.zeros((n_rx, Nc, Ns), dtype=np.complex128)

    for target in targets:
        R = target.range_m
        v = target.velocity
        rcs = target.rcs
        theta_az = np.deg2rad(target.azimuth_deg)
        phi_el = np.deg2rad(target.elevation_deg)

        if R > config.max_range:
            continue

        f_doppler = 2 * v * config.fc / c
        amplitude = np.sqrt(rcs)

        # Direction vector (trỏ về hướng mục tiêu)
        ux = np.sin(theta_az) * np.cos(phi_el)
        uy = np.sin(phi_el)
        uz = np.cos(theta_az) * np.cos(phi_el)
        direction_vec = np.array([ux, uy, uz])

        # Trong DDMA, tại RX r, tín hiệu là tổng của mọi TX t phát đồng thời
        for r_idx in range(n_rx):
            path_diff_rx = np.dot(rx_pos[r_idx], direction_vec)
            phase_rx = 2 * np.pi * path_diff_rx / config.wavelength

            for t_idx in range(n_tx):
                path_diff_tx = np.dot(tx_pos[t_idx], direction_vec)
                phase_tx = 2 * np.pi * path_diff_tx / config.wavelength

                # Tổng phase không gian của đường truyền: TX -> Target -> RX
                phase_spatial = phase_tx + phase_rx

                for m in range(Nc):
                    # 1. Cập nhật vị trí mục tiêu theo thời gian thực (Range Walk)
                    t_slow = m * config.t_chirp
                    R_m = R + v * t_slow

                    # 2. Beat frequency tại chirp m: Bị ảnh hưởng bởi khoảng cách hiện tại (R_m)
                    # và hiệu ứng Doppler trong fast-time (vận tốc làm xê dịch phổ beat)
                    f_beat_m = (2 * config.slope * R_m / c) + f_doppler

                    # Phase tích luỹ do Doppler (slow-time)
                    phase_slow = 2 * np.pi * f_doppler * t_slow

                    # Phase DDMA của TX ở chirp m
                    phase_ddma = ddma_phase[t_idx, m]

                    # 3. Bộ lọc chống hiện tượng chồng phổ (Anti-Aliasing Filter - AAF)
                    # Giả lập bộ lọc Analog Low-Pass Filter bậc 8 với tần số cắt f_c = f_s / 2
                    # Lọc bỏ các tín hiệu có tần số beat vượt quá giới hạn Nyquist
                    aaf_gain = 1.0 / \
                        np.sqrt(1 + (abs(f_beat_m) / (config.fs / 2))**(2 * 8))
                    current_amp = amplitude * aaf_gain

                    beat_signal[r_idx, m, :] += current_amp * np.exp(
                        1j * (2 * np.pi * f_beat_m * t_fast +
                              phase_slow + phase_spatial + phase_ddma)
                    )

    # Thêm nhiễu AWGN
    signal_power = np.mean(np.abs(beat_signal) ** 2)
    if signal_power > 0:
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_rx, Nc, Ns) + 1j * np.random.randn(n_rx, Nc, Ns)
        )
        beat_signal += noise

    return beat_signal


if __name__ == "__main__":
    pass
