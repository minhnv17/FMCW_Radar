"""
main.py — FMCW Radar Simulation Entry Point

Chạy toàn bộ pipeline mô phỏng FMCW Radar:
    1. Cấu hình tham số radar
    2. Định nghĩa mục tiêu (range, velocity, RCS)
    3. Tạo beat signal (IF signal) từ multi-target
    4. Xử lý tín hiệu: 2D FFT → Range-Doppler Map
    5. Hiển thị kết quả

Cách chạy:
    python main.py
"""

from radar_config import RadarConfig
from signal_generator import Target, gen_sig
from radar_types import DetectedTarget, RadarPoint, RadarPointCloud
from signal_processor import (
    compute_range_doppler_map,
    detect_targets_cfar,
)
from angle_estimator import (
    extract_virtual_array,
    compute_digital_beamforming,
)
from visualization import plot_results
import numpy as np


def main():
    # ================================================================
    # 1. Cấu hình radar
    # ================================================================
    # Tham số mặc định mô phỏng radar automotive 77 GHz
    # Có thể thay đổi bất kỳ tham số nào:
    #   RadarConfig(fc=24e9, bandwidth=200e6, ...)
    config = RadarConfig(
        fc=77e9,
        bandwidth=40e6,
        fs=40e6,
        n_chirps=512,
        n_samples=512,
    )

    # In ra thông số radar
    config.print_specs()

    # Hiển thị cấu trúc mảng ăng-ten ảo
    print("\n>> Viewing MIMO Antenna Array Layout...")
    # config.antenna_array.plot_array_layout() # Có thể uncomment để xem trực tiếp

    # ================================================================
    # 2. Định nghĩa mục tiêu
    # ================================================================
    # Mỗi target có: khoảng cách (m), vận tốc (m/s), RCS (m²)
    #   velocity > 0 → mục tiêu đang ra xa
    #   velocity < 0 → mục tiêu đang tiến lại gần
    targets = [
        Target(range_m=400,  velocity=9.3,   rcs=1,
               azimuth_deg=30),
        Target(range_m=402,  velocity=9.0, rcs=0.1, azimuth_deg=50),
        # Target(range_m=50,  velocity=0.0,  rcs=15),   # Xe cùng chiều, ra xa
        # Target(range_m=350,  velocity=0.0,  rcs=15),   # Xe cùng chiều, ra xa
    ]

    print(f"\nNumber of targets: {len(targets)}")
    for i, t in enumerate(targets):
        print(f"  Target {i+1}: range={t.range_m}m, "
              f"velocity={t.velocity:+.1f}m/s, RCS={t.rcs}")

    # ================================================================
    # 3. Tạo beat signal
    # ================================================================
    print("\n>> Generating beat signal with high noise (SNR = -5 dB)...")
    beat_signal = gen_sig(config, targets, snr_db=-10)
    print(f"  Beat signal shape: {beat_signal.shape} "
          f"(channel x chirps x samples)")

    # ================================================================
    # 4. Xử lý tín hiệu → Range-Doppler Map
    # ================================================================
    print(">> Computing Range-Doppler Map (2D FFT)...")
    doppler_fft, rdm_db, range_bins, doppler_bins = compute_range_doppler_map(
        config, beat_signal
    )
    print(f"  RDM shape: {rdm_db.shape}")
    print(f"  Range bins: {range_bins[0]} -> {range_bins[-1]}")
    print(f"  Doppler bins: {doppler_bins[0]} -> {doppler_bins[-1]}")

    print(">> Running 2D CA-CFAR & Peak Search...")
    det_doppler_idx, det_range_idx, threshold_map = detect_targets_cfar(
        rdm_db,
        threshold_offset_db=10.0
    )

    # List chứa thông tin mục tiêu phát hiện được
    detected_targets: list[DetectedTarget] = []
    print(f"  Detected {len(det_range_idx)} peaks:")
    # Tính toán vùng Baseband cho DDMA
    # Trong DDMA, phổ được chia làm n_tx dải.
    # Dải của TX0 (đỉnh thật) nằm quanh tần số 0, tức là từ bin 0 đến giới hạn Nyquist của DDMA,
    # và từ (Nc - giới hạn) đến (Nc - 1) đối với vận tốc âm.
    baseband_limit = config.n_chirps // (2 * config.n_tx)

    for d_idx, r_idx in zip(det_doppler_idx, det_range_idx):
        # Lọc DDMA: Bỏ qua các đỉnh nằm ngoài vùng Baseband
        if not (d_idx <= baseband_limit or d_idx >= config.n_chirps - baseband_limit):
            continue

        # Giải mã DDMA: Trích xuất mảng ảo
        v_arr, v_pos = extract_virtual_array(config, doppler_fft, r_idx, d_idx)

        # Tạo búp sóng số (Digital Beamforming) để đo góc
        spectrum_db, angles = compute_digital_beamforming(
            virtual_array=v_arr,
            virtual_pos=v_pos,
            wavelength=config.wavelength,
            angle_start=-60.0,
            angle_end=60.0,
            n_angles=121
        )

        # Theo yêu cầu: Peak chỉ được xác định bởi CFAR. DBF chỉ dùng để ước lượng 1 góc lớn nhất.
        max_angle_idx = int(np.argmax(spectrum_db))
        azimuth = float(angles[max_angle_idx])

        # Power thực của mục tiêu lấy từ Range-Doppler Map (không lấy từ góc vì phổ góc đã bị chuẩn hoá về 0)
        peak_power = float(rdm_db[r_idx, d_idx])

        # Convert indices to physical units
        r_val = float(r_idx * config.range_resolution)
        v_idx = d_idx if d_idx < config.n_chirps // 2 else d_idx - config.n_chirps
        v_val = float(v_idx * config.velocity_resolution)

        # Tính RCS tương đối từ peak_power (dB)
        rcs_val = float(10 ** (peak_power / 10))

        # Khởi tạo DetectedTarget
        dt = DetectedTarget(
            range_m=r_val,
            velocity=v_val,
            azimuth_deg=azimuth,
            power_db=peak_power,
            rcs=rcs_val,
            r_idx=r_idx,
            d_idx=d_idx,
            angle_spectrum=spectrum_db,
            angles=angles
        )

        detected_targets.append(dt)
        print(
            f"    - Range: {r_val:.1f} m, Velocity: {v_val:+.1f} m/s, Azimuth: {azimuth:+.1f}°")

    # Tạo Point Cloud
    point_cloud = RadarPointCloud(frame_id=1)
    for dt in detected_targets:
        point = RadarPoint(
            x=dt.x, y=dt.y, z=dt.z,
            velocity=dt.velocity,
            snr_db=dt.power_db,  # (Tạm dùng power làm SNR)
            power_db=dt.power_db,
            range_m=dt.range_m,
            azimuth_deg=dt.azimuth_deg,
            elevation_deg=dt.elevation_deg
        )
        point_cloud.points.append(point)

    print(
        f"    - Generated Point Cloud with {len(point_cloud.points)} points.")

    # ================================================================
    # 5. Hiển thị kết quả
    # ================================================================
    print(">> Plotting results...")
    plot_results(
        config=config,
        targets=targets,
        detected_targets=detected_targets,
        rdm_db=rdm_db,
        range_bins=range_bins,
        doppler_bins=doppler_bins,
        dynamic_range_db=40,
    )

    print("\n[OK] Simulation complete!")


if __name__ == "__main__":
    main()
