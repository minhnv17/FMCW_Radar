"""
signal_processor.py — Xử lý tín hiệu FMCW Radar

Module này thực hiện chuỗi xử lý tín hiệu (signal processing pipeline) để
chuyển đổi beat signal thô thành Range-Doppler Map.

Pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │  Beat Signal [Nc × Ns]                                      │
    │       │                                                     │
    │       ▼                                                     │
    │  ① Window (Hann) theo fast-time → giảm range sidelobes     │
    │       │                                                     │
    │       ▼                                                     │
    │  ② Range FFT (axis=1) → phổ tần số = range bins            │
    │       │                                                     │
    │       ▼                                                     │
    │  ③ Window (Hann) theo slow-time → giảm Doppler sidelobes   │
    │       │                                                     │
    │       ▼                                                     │
    │  ④ Doppler FFT (axis=0) → phổ Doppler = velocity bins      │
    │       │                                                     │
    │       ▼                                                     │
    │  ⑤ fftshift + magnitude → Range-Doppler Map [dB]           │
    │       │                                                     │
    │       ▼                                                     │
    │  Output: rdm_db, range_axis, velocity_axis                  │
    └─────────────────────────────────────────────────────────────┘

Tại sao cần windowing?
    FFT giả định tín hiệu tuần hoàn. Nếu tín hiệu bị cắt đột ngột ở biên,
    sẽ xuất hiện "spectral leakage" — năng lượng bị rò rỉ sang các bin lân cận,
    tạo ra sidelobes giả. Cửa sổ Hann giảm biên tín hiệu về 0 mượt mà,
    giảm sidelobes ~31 dB so với không dùng window.
"""

import scipy.ndimage as ndimage
import numpy as np

from radar_config import RadarConfig, SPEED_OF_LIGHT


def compute_range_fft(
    config: RadarConfig,
    beat_signal: np.ndarray,
) -> np.ndarray:
    """
    Thực hiện Range FFT (FFT theo fast-time dimension).

    Áp dụng Hann window trước khi FFT để giảm spectral leakage.

    FFT biến đổi tín hiệu beat từ miền thời gian sang miền tần số.
    Mỗi bin tần số tương ứng với 1 range bin:
        Range[k] = k × ΔR = k × c / (2B)

    Args:
        config:       Cấu hình radar.
        beat_signal:  Ma trận beat signal [Nc × Ns].

    Returns:
        range_fft:    Ma trận sau Range FFT [Nc × Ns] (phức).
    """
    Ns = config.n_samples

    # Tạo Hann window cho fast-time dimension
    # Window shape: (1, 1, Ns) để broadcast với ma trận (N_virtual, Nc, Ns)
    window = np.hanning(Ns).reshape(1, 1, -1)

    # Áp dụng window và FFT theo axis=-1 (fast-time)
    windowed = beat_signal * window
    range_fft = np.fft.fft(windowed, axis=-1)

    return range_fft


def compute_doppler_fft(
    config: RadarConfig,
    range_fft: np.ndarray,
) -> np.ndarray:
    """
    Thực hiện Doppler FFT (FFT theo slow-time dimension).

    Sau Range FFT, mỗi cột chứa giá trị phức của cùng 1 range bin
    qua các chirp liên tiếp. Phase của các giá trị này thay đổi theo
    vận tốc mục tiêu:
        Δφ = 2π x f_d x T_c = 4π x v x T_c / λ

    FFT theo slow-time (axis=0) giải mã phase này thành Doppler bins.

    Args:
        config:     Cấu hình radar.
        range_fft:  Ma trận sau Range FFT [Nc x Ns].

    Returns:
        doppler_fft: Ma trận sau 2D FFT [Nc x Ns] (phức).
    """
    Nc = config.n_chirps

    # Tạo Hann window cho slow-time dimension
    # Window shape: (1, Nc, 1) để broadcast
    window = np.hanning(Nc).reshape(1, -1, 1)

    # Áp dụng window và FFT theo axis=1 (slow-time, tức là dimension của Nc)
    windowed = range_fft * window
    doppler_fft = np.fft.fft(windowed, axis=1)

    return doppler_fft


def compute_range_doppler_map(
    config: RadarConfig,
    beat_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính toán Range-Doppler Map hoàn chỉnh từ beat signal.

    Đây là hàm chính kết hợp toàn bộ pipeline:
        Beat signal → Range FFT → Doppler FFT → |·|² → dB → fftshift

    Args:
        config:       Cấu hình radar.
        beat_signal:  Ma trận beat signal [Nc × Ns].

    Returns:
        doppler_fft:  Ma trận 2D FFT phức [n_rx × Nc × Ns].
        rdm_db:       Range-Doppler Map [Nc × Ns/2] (dB, normalized).
        range_bins:   Chỉ số range bin [0, 1, ..., Ns/2-1].
        doppler_bins: Chỉ số doppler bin [-Nc/2, ..., Nc/2-1].
    """
    Nc = config.n_chirps
    Ns = config.n_samples

    # ── Bước 1 & 2: Range FFT ──────────────────────────────────
    range_fft = compute_range_fft(config, beat_signal)

    # ── Bước 3 & 4: Doppler FFT ────────────────────────────────
    doppler_fft = compute_doppler_fft(config, range_fft)

    # ── Bước 5: Post-processing ────────────────────────────────

    # Non-coherent integration: Tính tổng công suất qua tất cả các ăng-ten ảo
    # doppler_fft có kích thước (n_virtual, Nc, Ns)
    # Kích thước còn lại: (Nc, Ns)
    rdm_power = np.sum(np.abs(doppler_fft)**2, axis=0)

    # Chỉ lấy nửa đầu range (bins 0 → Ns/2):
    rdm_power = rdm_power[:, :Ns // 2]

    # Power → dB
    rdm_power[rdm_power == 0] = 1e-12  # tránh log(0)
    rdm_db = 10 * np.log10(rdm_power)

    # Normalize: peak = 0 dB
    rdm_db -= np.max(rdm_db)

    # ── Tạo trục bin ────────────────────────────────────────────
    range_bins = np.arange(Ns // 2)
    doppler_bins = np.arange(Nc)

    return doppler_fft, rdm_db, range_bins, doppler_bins


def detect_targets_cfar(
    rdm_db: np.ndarray,
    train_cells_range: int = 4,
    train_cells_doppler: int = 4,
    guard_cells_range: int = 2,
    guard_cells_doppler: int = 2,
    threshold_offset_db: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phát hiện mục tiêu bằng thuật toán 2D CA-CFAR (Cell-Averaging CFAR) kết hợp Peak Search.

    Thuật toán:
    1. Quét một cửa sổ (Window) 2D qua ma trận RDM.
    2. Window gồm: Cell đang xét (CUT - Cell Under Test), Guard cells (chống rò rỉ tín hiệu),
       và Training cells (để ước lượng nhiễu nền).
    3. Ngưỡng động = Trung bình(Training cells) + threshold_offset_db.
    4. Chỉ giữ lại những cell có biên độ > Ngưỡng động.
    5. Kết hợp Peak Search (Local Maxima) để chỉ lấy 1 đỉnh duy nhất cho mỗi mục tiêu,
       loại bỏ các cell lân cận bị lan ra do nhiễu.

    Returns:
        doppler_indices: Mảng chứa index của doppler bin của các mục tiêu.
        range_indices: Mảng chứa index của range bin của các mục tiêu.
        cfar_threshold_map: Ma trận ngưỡng động (để debug/hiển thị nếu cần).
    """
    # ── 1. Tạo Kernel cho 2D CFAR ────────────────────────────────
    # Kích thước toàn bộ window
    win_r = 2 * (train_cells_range + guard_cells_range) + 1
    win_d = 2 * (train_cells_doppler + guard_cells_doppler) + 1

    # Kích thước vùng cấm (CUT + Guard cells)
    guard_r = 2 * guard_cells_range + 1
    guard_d = 2 * guard_cells_doppler + 1

    # Ma trận kernel (1 cho training cells, 0 cho phần còn lại)
    kernel = np.ones((win_d, win_r))

    # Đặt vùng giữa (CUT + Guard) thành 0
    start_d = train_cells_doppler
    end_d = start_d + guard_d
    start_r = train_cells_range
    end_r = start_r + guard_r
    kernel[start_d:end_d, start_r:end_r] = 0

    # Chuẩn hoá kernel để tính trung bình
    num_train_cells = np.sum(kernel)
    kernel = kernel / num_train_cells

    # ── 2. Ước lượng nhiễu nền (Noise Floor) ────────────────────
    # Dùng convolution (hoặc filter) để trượt kernel qua toàn bộ RDM cực nhanh
    # rdm_db đang ở thang log (dB), nên đây là thuật toán Log-CA-CFAR
    noise_floor_db = ndimage.convolve(rdm_db, kernel, mode='reflect')

    # ── 3. Tính ngưỡng (Threshold) ──────────────────────────────
    cfar_threshold_map = noise_floor_db + threshold_offset_db

    # ── 4. Tạo Boolean Mask (Cell > Threshold) ──────────────────
    is_target_cfar = rdm_db > cfar_threshold_map

    # ── 5. Kết hợp Local Peak Search ────────────────────────────
    # Tìm local maxima trong vùng 3x3 để tránh nhận diện nhiều ô cho 1 mục tiêu
    local_max = ndimage.maximum_filter(rdm_db, size=3)
    is_peak = (rdm_db == local_max)

    # Mục tiêu cuối cùng = phải là Đỉnh (Peak) VÀ vượt ngưỡng CFAR
    final_detection_mask = is_target_cfar & is_peak

    # Lấy ra toạ độ của các mục tiêu
    # argwhere trả về mảng shape (N, 2) với [doppler_idx, range_idx]
    target_indices = np.argwhere(final_detection_mask)

    if len(target_indices) > 0:
        doppler_indices = target_indices[:, 0]
        range_indices = target_indices[:, 1]
    else:
        doppler_indices = np.array([])
        range_indices = np.array([])

    return doppler_indices, range_indices, cfar_threshold_map


def compute_single_range_profile(
    config: RadarConfig,
    beat_signal: np.ndarray,
    chirp_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính range profile cho 1 chirp duy nhất.

    Hữu ích để kiểm tra nhanh: peak trong range profile phải xuất hiện
    tại vị trí tương ứng khoảng cách mục tiêu.

    Args:
        config:       Cấu hình radar.
        beat_signal:  Ma trận beat signal [Nc × Ns].
        chirp_index:  Chỉ số chirp cần phân tích (mặc định = 0).

    Returns:
        range_profile_db:  |FFT|² [dB], shape (Ns/2,).
        range_axis:        Trục range [m], shape (Ns/2,).
    """
    Ns = config.n_samples

    # Lấy 1 chirp của ăng-ten ảo đầu tiên
    chirp_data = beat_signal[0, chirp_index, :]

    # Windowing + FFT
    window = np.hanning(Ns)
    spectrum = np.fft.fft(chirp_data * window)

    # Lấy nửa đầu, chuyển sang dB
    half = Ns // 2
    magnitude = np.abs(spectrum[:half])
    magnitude[magnitude == 0] = 1e-12
    profile_db = 20 * np.log10(magnitude)
    profile_db -= np.max(profile_db)  # Normalize

    # Range axis
    range_axis = np.arange(half) * config.range_resolution

    return profile_db, range_axis
