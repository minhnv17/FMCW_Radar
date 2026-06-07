"""
angle_estimator.py — Giải mã DDMA và Ước lượng Góc

Module này thực hiện hai nhiệm vụ chính của Radar MIMO:
1. Giải mã DDMA: Trích xuất các tín hiệu từ 4 đỉnh Doppler khác nhau để ghép lại
   thành một mảng ăng-ten ảo (Virtual Array).
2. Digital Beamforming (DBF): Áp dụng thuật toán Delay & Sum để ước lượng góc 
   Azimuth của mục tiêu từ mảng ăng-ten ảo không đồng đều.
"""

import numpy as np
from radar_config import RadarConfig

def extract_virtual_array(
    config: RadarConfig,
    doppler_fft: np.ndarray,
    r_idx: int,
    d_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trích xuất mảng ăng-ten ảo từ bản đồ Doppler cho một mục tiêu cụ thể.
    
    Args:
        config: Cấu hình Radar
        doppler_fft: Ma trận FFT phức (n_rx, Nc, Ns)
        r_idx: Chỉ số Range bin của mục tiêu
        d_idx: Chỉ số Doppler bin gốc (Baseband) của mục tiêu
        
    Returns:
        virtual_array: Mảng giá trị phức (complex) của các phần tử ăng-ten ảo.
        virtual_pos: Toạ độ tương ứng của các phần tử ăng-ten ảo.
    """
    n_rx = config.n_rx
    n_tx = config.n_tx
    Nc = config.n_chirps
    
    # Số lượng ăng-ten ảo là n_tx * n_rx
    n_virtual = config.n_virtual
    virtual_array = np.zeros(n_virtual, dtype=np.complex128)
    
    # Khoảng cách giữa các đỉnh Doppler do DDMA sinh ra
    peak_offset = Nc // n_tx
    
    for t_idx in range(n_tx):
        # Tính vị trí Doppler bin bị dịch đi do mã pha của TX tương ứng
        d_shifted = (d_idx + t_idx * peak_offset) % Nc
        
        # Rút trích giá trị phức tại r_idx và d_shifted cho TẤT CẢ các ăng-ten thu (RX)
        rx_values = doppler_fft[:, d_shifted, r_idx]
        
        # Ánh xạ vào mảng ăng-ten ảo
        # AntennaArray tạo virtual_pos bằng vòng lặp 2 lớp: duyệt TX, sau đó duyệt RX
        start_idx = t_idx * n_rx
        end_idx = start_idx + n_rx
        virtual_array[start_idx:end_idx] = rx_values
        
    assert config.antenna_array is not None
    virtual_pos = config.antenna_array.virtual_pos
    return virtual_array, virtual_pos


def compute_digital_beamforming(
    virtual_array: np.ndarray,
    virtual_pos: np.ndarray,
    wavelength: float,
    angle_start: float = -90.0,
    angle_end: float = 90.0,
    n_angles: int = 181
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính toán phổ không gian (Angle Spectrum) bằng Digital Beamforming (Delay & Sum).
    Hỗ trợ mọi cấu trúc mảng ăng-ten kể cả không đồng đều (non-ULA).
    
    Args:
        virtual_array: Mảng tín hiệu phức (n_virtual,)
        virtual_pos: Ma trận toạ độ các ăng-ten ảo (n_virtual, 3)
        wavelength: Bước sóng radar (m)
        angle_start, angle_end: Giới hạn quét góc (độ)
        n_angles: Số lượng điểm quét (độ phân giải)
        
    Returns:
        spectrum_db: Mảng độ lớn phổ không gian (dB), giá trị đỉnh là 0 dB.
        angles: Trục góc quét (độ).
    """
    angles = np.linspace(angle_start, angle_end, n_angles)
    theta_rad = np.deg2rad(angles)
    
    # Vector hướng (Direction Vector) giả định tia quét nằm trên mặt phẳng nằm ngang (Elevation=0)
    # ux = sin(theta), uy = 0, uz = cos(theta)
    ux = np.sin(theta_rad)
    
    # Trích xuất toạ độ X của các ăng-ten ảo (trục Azimuth)
    x_pos = virtual_pos[:, 0]
    
    # Tính ma trận bù pha (Steering Matrix)
    # Kích thước: (n_angles, n_virtual)
    # Công thức: A = exp(-j * 2*pi * x_pos * sin(theta) / lambda)
    steering_matrix = np.exp(-1j * 2 * np.pi * np.outer(ux, x_pos) / wavelength)
    
    # Thực hiện chùm tia số (Digital Beamforming): Nhân ma trận bù pha với mảng ảo
    beam_output = np.dot(steering_matrix, virtual_array)
    
    # Chuyển sang công suất và biểu diễn theo dB
    power = np.abs(beam_output) ** 2
    power[power == 0] = 1e-12
    spectrum_db = 10 * np.log10(power)
    
    # Chuẩn hoá để đỉnh cao nhất bằng 0 dB
    spectrum_db -= np.max(spectrum_db)
    
    return spectrum_db, angles
