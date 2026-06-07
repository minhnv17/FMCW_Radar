import numpy as np
from dataclasses import dataclass, field

@dataclass
class DetectedTarget:
    """
    Thông tin về một mục tiêu đã được Radar (CFAR + DBF) phát hiện.
    Được sử dụng cho Visualization và các thuật toán Tracking (Kalman Filter).
    """
    # Toạ độ cực (Polar)
    range_m: float          # Khoảng cách [m]
    velocity: float         # Vận tốc hướng tâm [m/s]
    azimuth_deg: float      # Góc phương vị ngang [độ]
    elevation_deg: float = 0.0 # Góc tà dọc [độ]
    
    # Năng lượng
    power_db: float = 0.0   # Năng lượng tại đỉnh RDM [dB]
    rcs: float = 0.0        # Radar Cross Section tương đối
    snr_db: float = 0.0     # Signal-to-Noise Ratio [dB] (tuỳ chọn)

    # Thông tin debug (chỉ dùng cho hiển thị / R&D)
    r_idx: int = 0
    d_idx: int = 0
    from typing import Optional
    angle_spectrum: Optional[np.ndarray] = field(default=None, repr=False)
    angles: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def x(self) -> float:
        """Toạ độ X (m) trong hệ toạ độ Cartesian."""
        return self.range_m * np.sin(np.deg2rad(self.azimuth_deg)) * np.cos(np.deg2rad(self.elevation_deg))

    @property
    def y(self) -> float:
        """Toạ độ Y (m) - Hướng nhìn thẳng của Radar."""
        return self.range_m * np.cos(np.deg2rad(self.azimuth_deg)) * np.cos(np.deg2rad(self.elevation_deg))

    @property
    def z(self) -> float:
        """Toạ độ Z (m) - Chiều cao."""
        return self.range_m * np.sin(np.deg2rad(self.elevation_deg))


@dataclass
class RadarPoint:
    """Một điểm (point) trong Point Cloud."""
    x: float
    y: float
    z: float
    velocity: float
    snr_db: float
    power_db: float
    range_m: float
    azimuth_deg: float
    elevation_deg: float


@dataclass
class RadarPointCloud:
    """
    Cấu trúc dữ liệu lưu trữ Point Cloud xuất ra từ Radar.
    Tương tự cấu trúc PointCloud2 trong ROS (Robot Operating System) 
    hoặc Texas Instruments (TI) mmWave OOB format.
    """
    timestamp: float = 0.0
    frame_id: int = 0
    points: list[RadarPoint] = field(default_factory=list)

    def to_numpy(self) -> np.ndarray:
        """
        Trích xuất Point Cloud thành mảng numpy (N x 5) để xử lý nhanh bằng thư viện ngoài.
        Cột: [x, y, z, velocity, snr]
        """
        if not self.points:
            return np.empty((0, 5))
            
        data = []
        for p in self.points:
            data.append([p.x, p.y, p.z, p.velocity, p.snr_db])
        return np.array(data)
