# FMCW MIMO Radar Simulation — Python Research Project

Dự án mô phỏng **FMCW (Frequency Modulated Continuous Wave) Radar** bằng Python, hỗ trợ kiến trúc **MIMO (Multiple-Input Multiple-Output)** với kỹ thuật mã hóa **DDMA (Doppler Division Multiple Access)**. 

Dự án mô phỏng toàn bộ chuỗi xử lý tín hiệu số (DSP) từ việc tạo tín hiệu thô (beat signal) cho đến trích xuất đám mây điểm (Point Cloud) 3D, ứng dụng cho xe tự lái (Automotive Radar) hoặc theo dõi mục tiêu.

## Tính năng nổi bật

1. **MIMO & DDMA**: Hỗ trợ nhiều ăng-ten thu phát (Mặc định: ULA 4 TX x 4 RX = 16 Virtual Elements). Mã hóa trực giao DDMA giúp tách biệt tín hiệu của các TX trên miền Doppler.
2. **2D CA-CFAR**: Thuật toán Cell-Averaging Constant False Alarm Rate phát hiện mục tiêu động tự động trên Range-Doppler Map, loại bỏ nhiễu linh hoạt.
3. **Digital Beamforming (DBF)**: Ước lượng góc phương vị (Azimuth Angle of Arrival) dựa trên mảng ăng-ten ảo.
4. **Data Structures Chuẩn Công Nghiệp**: Xuất dữ liệu dưới dạng `RadarPointCloud` (tương tự định dạng ROS hoặc TI mmWave).

## Cấu trúc dự án

```
FMCWRadar/
├── main.py                # Entry point — chạy toàn bộ pipeline
├── radar_config.py        # Tham số radar (dataclass)
├── radar_types.py         # Định nghĩa cấu trúc dữ liệu OOP (DetectedTarget, RadarPointCloud)
├── antenna_array.py       # Cấu trúc mảng ăng-ten MIMO (ULA, L-Shape, NiDAR...)
├── signal_generator.py    # Tạo chirp TX, beat signal RX (multi-target) với Phase Shift
├── signal_processor.py    # Range FFT, Doppler FFT → Range-Doppler Map & CFAR
├── angle_estimator.py     # Giải mã DDMA và Digital Beamforming (DBF)
├── visualization.py       # Render hiển thị Dashboard kết quả (5-panel figure)
├── requirements.txt       # Dependencies
└── README.md              # File này
```

## Cài đặt

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Cài đặt dependencies
pip install -r requirements.txt
```

## Chạy Demo

```bash
python main.py
```

## Kết quả hiển thị (Dashboard)

Chương trình hiển thị 5 panel trực quan:

| Panel | Nội dung |
|-------|----------|
| **Doppler Profile** | Phân bố năng lượng theo vận tốc |
| **Range Profile** | Phân bố năng lượng theo khoảng cách |
| **Range-Doppler Map** | Heatmap 2D (Range × Velocity) cùng các điểm CFAR khoanh vùng |
| **Radar Specs** | Bảng thông số Radar và Danh sách mục tiêu phát hiện được |
| **Angle Spectrum** | Đồ thị toạ độ cực (Polar Plot) hiển thị búp sóng DBF và góc Azimuth |

## Tham số mặc định (Automotive Radar)

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| f_c | 77.0 GHz | Tần số sóng mang (automotive band) |
| B | 40.0 MHz | Băng thông chirp |
| T_c | 25.6 µs | Thời gian 1 chirp |
| N_chirps | 512 | Số chirp / frame |
| N_samples | 1024 | Số mẫu ADC / chirp |
| ΔR | ~3.75 m | Range resolution |
| v_max | ±9.51 m/s | Max velocity limit |
| MIMO | 4 TX, 4 RX | Uniform Linear Array (ULA) - Không có Grating Lobes |

## Triết lý Thiết kế (Architecture)

Dự án áp dụng mô hình **CFAR First, DBF Last**:
1. **Detection (Phát hiện)**: Giao phó hoàn toàn cho khối 2D CA-CFAR trên bản đồ Range-Doppler.
2. **Estimation (Đo góc)**: Khối DBF chỉ được "đánh thức" tại đúng những toạ độ mà CFAR chỉ định, giúp triệt tiêu hoàn toàn Ghost Targets và tiết kiệm tính toán.

## Mở rộng

Dự án được thiết kế modular, bạn có thể dễ dàng phát triển thêm:
- **Kalman Tracking**: Đưa Point Cloud vào bộ lọc Kalman để theo dõi quỹ đạo (Trajectories).
- **ROS / Rviz Integration**: Đẩy Point Cloud ra qua ROS topics.
- **Hardware Integration**: Thay thế module `signal_generator.py` bằng API đọc dữ liệu raw từ board mạch thật (TI IWR6843, Infineon).
