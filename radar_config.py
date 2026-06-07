"""
radar_config.py — Cấu hình tham số FMCW Radar

Module này định nghĩa dataclass chứa toàn bộ tham số radar và các thông số
dẫn xuất (range resolution, max range, velocity resolution, max velocity).

Lý thuyết cơ bản:
    FMCW Radar phát tín hiệu chirp — sóng có tần số tăng tuyến tính từ f_c
    đến f_c + B trong khoảng thời gian T_c. Khi tín hiệu phản xạ về, nó bị
    trễ τ = 2R/c. Trộn (mix) tín hiệu phát và nhận tạo ra beat frequency:
        f_b = S × τ = (2 × S × R) / c
    với S = B / T_c là chirp slope (Hz/s).
"""

import numpy as np
from dataclasses import dataclass, field
from antenna_array import AntennaArray
from typing import Optional

# ============================================================================
# Hằng số vật lý
# ============================================================================
SPEED_OF_LIGHT = 3e8  # m/s


@dataclass
class RadarConfig:
    """
    Tham số cấu hình FMCW Radar.

    Attributes:
        fc:         Tần số sóng mang (Hz). Mặc định 77 GHz — dải automotive.
        bandwidth:  Băng thông chirp B (Hz). Quyết định range resolution.
        fs:         Tần số lấy mẫu ADC (Hz).
        n_chirps:   Số chirp trong 1 frame (slow-time dimension).
        n_samples:  Số mẫu ADC mỗi chirp (fast-time dimension).
    """

    # --- Tham số chính (có thể thay đổi) ---
    fc: float = 77e9            # Carrier frequency [Hz]
    bandwidth: float = 150e6    # Chirp bandwidth B [Hz]
    fs: float = 51.2e6          # ADC sampling frequency [Hz]
    n_chirps: int = 512         # Số chirp / frame
    n_samples: int = 1024        # Số mẫu ADC / chirp

    # --- Tham số dẫn xuất (tự tính) ---
    # Chirp slope S = B / T_c [Hz/s]
    # Thời gian 1 chirp T_c = N / fs [s]
    t_chirp: float = field(init=False)
    slope: float = field(init=False)
    wavelength: float = field(init=False)        # Bước sóng λ = c / f_c [m]
    range_resolution: float = field(init=False)  # ΔR = c / (2B) [m]
    max_range: float = field(init=False)         # R_max [m]
    velocity_resolution: float = field(init=False)  # Δv [m/s]
    max_velocity: float = field(init=False)      # v_max [m/s]
    angle_resolution: float = field(init=False)  # Δθ [deg]
    max_angle: float = field(init=False)         # θ_max [deg]
    antenna_array: Optional[AntennaArray] = None

    def __post_init__(self):
        """Tính toán các tham số dẫn xuất từ tham số chính."""
        c = SPEED_OF_LIGHT

        # Khởi tạo mảng ăng-ten MIMO dạng L mặc định nếu người dùng không cung cấp
        if self.antenna_array is None:
            # Tính tạm wavelength để khởi tạo antenna_array
            temp_wavelength = c / self.fc
            from antenna_array import create_uniform_linear_mimo_4x4
            self.antenna_array = create_uniform_linear_mimo_4x4(
                temp_wavelength)

        # Thời gian 1 chirp (Chirp duration)
        #   T_c = N_samples / f_s  [s]
        self.t_chirp = self.n_samples / self.fs

        # Chirp slope: tốc độ thay đổi tần số
        #   S = B / T_c  [Hz/s]
        self.slope = self.bandwidth / self.t_chirp

        # Bước sóng tại tần số sóng mang
        #   λ = c / f_c  [m]
        self.wavelength = c / self.fc

        # --- Range ---
        # Range resolution: khả năng phân biệt 2 mục tiêu gần nhau
        #   ΔR = c / (2 × B)  [m]
        self.range_resolution = c / (2 * self.bandwidth)

        # Maximum unambiguous range
        #   Do ta chỉ lấy nửa phổ dương (Ns/2) cho tín hiệu thực:
        #   R_max = (N_samples/2) × ΔR = (f_s × c) / (4 × S)
        self.max_range = (self.fs * c) / (4 * self.slope)

        # --- Velocity ---
        # Velocity resolution: khả năng phân biệt 2 vận tốc
        #   Δv = λ / (2 × N_chirps × T_c)  [m/s]
        self.velocity_resolution = self.wavelength / \
            (2 * self.n_chirps * self.t_chirp)

        # Maximum unambiguous velocity
        #   Trong MIMO DDMA, phổ Doppler bị chia làm N_tx phần
        #   v_max = λ / (4 × T_c × N_tx)  [m/s]
        self.max_velocity = self.wavelength / (4 * self.t_chirp * self.n_tx)

        # --- Angle ---
        # Mảng ảo ULA với d = λ/2
        d = self.wavelength / 2
        # Max angle (Field of View) = ± arcsin(λ / (2d))
        self.max_angle = np.degrees(np.arcsin(self.wavelength / (2 * d)))
        # Angle resolution = λ / (N_virtual * d)
        self.angle_resolution = np.degrees(
            self.wavelength / (self.n_virtual * d))

    @property
    def n_tx(self) -> int:
        assert self.antenna_array is not None
        return self.antenna_array.n_tx

    @property
    def n_rx(self) -> int:
        assert self.antenna_array is not None
        return self.antenna_array.n_rx

    @property
    def n_virtual(self) -> int:
        assert self.antenna_array is not None
        return self.antenna_array.n_virtual

    def print_specs(self):
        """In ra toàn bộ thông số và giới hạn của radar."""
        print("=" * 55)
        print("        FMCW RADAR CONFIGURATION")
        print("=" * 55)
        print(f"  Carrier frequency  : {self.fc / 1e9:.1f} GHz")
        print(f"  Bandwidth          : {self.bandwidth / 1e6:.1f} MHz")
        print(f"  Chirp duration     : {self.t_chirp * 1e6:.1f} us")
        print(f"  Chirp slope        : {self.slope / 1e12:.2f} MHz/us")
        print(f"  Wavelength         : {self.wavelength * 1e3:.2f} mm")
        print(f"  Sampling frequency : {self.fs / 1e6:.2f} MHz")
        print(f"  Samples per chirp  : {self.n_samples}")
        print(f"  Chirps per frame   : {self.n_chirps}")
        print("-" * 55)
        print(f"  Range resolution   : {self.range_resolution:.2f} m")
        print(f"  Max range          : {self.max_range:.1f} m")
        print(f"  Velocity resolution: {self.velocity_resolution:.2f} m/s")
        print(f"  Max velocity       : {self.max_velocity:.2f} m/s")
        print("-" * 55)
        print(f"  Antenna TX         : {self.n_tx}")
        print(f"  Antenna RX         : {self.n_rx}")
        print(f"  Virtual Elements   : {self.n_virtual}")
        print("=" * 55)

    def get_specs_text(self) -> str:
        """Trả về thông số radar dưới dạng chuỗi (cho visualization)."""
        lines = [
            "FMCW RADAR SPECS",
            "─" * 32,
            f"Carrier freq   : {self.fc / 1e9:.1f} GHz",
            f"Bandwidth      : {self.bandwidth / 1e6:.1f} MHz",
            f"Chirp duration : {self.t_chirp * 1e6:.1f} us",
            f"Chirp slope    : {self.slope / 1e12:.2f} MHz/us",
            f"Wavelength     : {self.wavelength * 1e3:.2f} mm",
            f"Sampling freq  : {self.fs / 1e6:.2f} MHz",
            f"Samples/chirp  : {self.n_samples}",
            f"Chirps/frame   : {self.n_chirps}",
            "",
            "PERFORMANCE",
            "─" * 32,
            f"Range res      : {self.range_resolution:.2f} m",
            f"Max range      : {self.max_range:.1f} m",
            f"Velocity res   : {self.velocity_resolution:.2f} m/s",
            f"Max velocity   : {self.max_velocity:.2f} m/s",
            "",
            "ANTENNA ARRAY",
            "─" * 32,
            f"TX Antennas    : {self.n_tx}",
            f"RX Antennas    : {self.n_rx}",
            f"Virtual Array  : {self.n_virtual} elements",
        ]
        return "\n".join(lines)
