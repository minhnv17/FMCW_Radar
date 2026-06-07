"""
antenna_array.py — Module quản lý cấu hình mảng ăng-ten MIMO

Chứa lớp quản lý vị trí các ăng-ten phát (TX), thu (RX), và tính toán mảng ăng-ten ảo (Virtual Array).
Cung cấp các hàm hỗ trợ khởi tạo nhanh một số cấu hình phổ biến (ví dụ: L-shape array)
và hiển thị bố cục ăng-ten trực quan.
"""

import numpy as np
import matplotlib.pyplot as plt


class AntennaArray:
    """
    Quản lý cấu hình mảng ăng-ten vật lý (TX, RX) và mảng ảo (Virtual Array).
    Tọa độ sử dụng hệ trục (x, y, z), trong đó:
        - x: Trục ngang (Azimuth)
        - y: Trục dọc (Elevation)
        - z: Hướng lan truyền sóng (thường là 0 cho mảng phẳng 2D)
    """

    def __init__(self, tx_positions: np.ndarray, rx_positions: np.ndarray):
        """
        Khởi tạo mảng ăng-ten.

        Args:
            tx_positions: Ma trận (N_tx, 3) tọa độ các ăng-ten phát [m]
            rx_positions: Ma trận (N_rx, 3) tọa độ các ăng-ten thu [m]
        """
        self.tx_pos = np.asarray(tx_positions)
        self.rx_pos = np.asarray(rx_positions)

        # Đảm bảo mảng có số chiều đúng (N, 3)
        if self.tx_pos.shape[1] != 3 or self.rx_pos.shape[1] != 3:
            raise ValueError(
                "Tọa độ TX và RX phải là mảng 2 chiều kích thước (N, 3)")

        self.virtual_pos = self._compute_virtual_array()

    def _compute_virtual_array(self) -> np.ndarray:
        """
        Tính toán mảng ăng-ten ảo.
        Với MIMO FMCW sử dụng TDM (Time Division Multiplexing) hoặc dải tần chuẩn trực giao,
        vị trí phần tử ảo = vị trí TX + vị trí RX.
        """
        virtual = []
        for tx in self.tx_pos:
            for rx in self.rx_pos:
                virtual.append(tx + rx)
        return np.array(virtual)

    @property
    def n_tx(self) -> int:
        """Số lượng ăng-ten phát"""
        return len(self.tx_pos)

    @property
    def n_rx(self) -> int:
        """Số lượng ăng-ten thu"""
        return len(self.rx_pos)

    @property
    def n_virtual(self) -> int:
        """Số lượng phần tử trong mảng ăng-ten ảo"""
        return len(self.virtual_pos)

    def plot_array_layout(self):
        """
        Trực quan hóa layout của TX, RX và Virtual Array trên mặt phẳng XY.
        Giả định tất cả ăng-ten nằm trên mặt phẳng z=0.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
        fig.suptitle("MIMO Antenna Array Layout", fontsize=16,
                     color="#e0e0ff", fontweight="bold")

        # Màu sắc chung
        face_color = "#16213e"
        text_color = "#b0b0d0"
        title_color = "#e0e0ff"
        grid_color = "#4a4a6a"

        # 1. Physical array (TX & RX)
        ax1 = axes[0]
        ax1.set_facecolor(face_color)
        ax1.scatter(self.tx_pos[:, 0], self.tx_pos[:, 1], c='#ff4757', marker='^',
                    s=150, label=f'TX ({self.n_tx})', edgecolor='white', linewidth=0.5)
        ax1.scatter(self.rx_pos[:, 0], self.rx_pos[:, 1], c='#1e90ff', marker='o',
                    s=80, label=f'RX ({self.n_rx})', edgecolor='white', linewidth=0.5)

        # Đánh số thứ tự TX, RX
        for i, pos in enumerate(self.tx_pos):
            ax1.annotate(f'T{i}', (pos[0], pos[1]), xytext=(
                5, 5), textcoords='offset points', color='#ff4757', fontsize=9)
        for i, pos in enumerate(self.rx_pos):
            ax1.annotate(f'R{i}', (pos[0], pos[1]), xytext=(
                5, -12), textcoords='offset points', color='#1e90ff', fontsize=9)

        ax1.set_title("Physical Array (TX & RX)", color=title_color)
        ax1.set_xlabel("X (Azimuth) [m]", color=text_color)
        ax1.set_ylabel("Y (Elevation) [m]", color=text_color)
        ax1.legend(loc='upper right', facecolor=face_color,
                   edgecolor=grid_color, labelcolor=text_color)
        ax1.grid(color=grid_color, alpha=0.3)
        ax1.tick_params(colors=text_color)
        ax1.axis('equal')

        # 2. Virtual array
        ax2 = axes[1]
        ax2.set_facecolor(face_color)

        # Gom nhóm các vị trí ảo bị trùng lặp (tránh text/điểm bị dính vào nhau)
        # Sử dụng round để tránh sai số floating point
        rounded_pos = np.round(self.virtual_pos, decimals=5)
        unique_pos, inverse_indices, counts = np.unique(
            rounded_pos, axis=0, return_inverse=True, return_counts=True)

        # Lấy danh sách các V_idx cho mỗi vị trí unique
        v_idx_groups = [[] for _ in range(len(unique_pos))]
        for v_idx, unq_idx in enumerate(inverse_indices):
            v_idx_groups[unq_idx].append(v_idx)

        # Vẽ các điểm unique, kích thước điểm (s) tỉ lệ với số lượng trùng lặp
        sizes = 100 + (counts - 1) * 80
        ax2.scatter(unique_pos[:, 0], unique_pos[:, 1], c='#ffa502', marker='*', s=sizes,
                    label=f'Virtual ({self.n_virtual} total, {len(unique_pos)} unique)',
                    edgecolor='white', linewidth=0.8, alpha=0.9)

        # Đánh dấu số lượng (Weight) và danh sách V_idx tại mỗi điểm
        for i, pos in enumerate(unique_pos):
            count = counts[i]
            # Nếu ít điểm (<=2) thì in ra V0, V1. Nếu nhiều thì in ra (x4)
            if count <= 2:
                label_text = ",".join([f"V{idx}" for idx in v_idx_groups[i]])
            else:
                label_text = f"(x{count})"

            ax2.annotate(label_text, (pos[0], pos[1]), xytext=(0, 10),
                         textcoords='offset points', color='#ffcccc', fontsize=9,
                         ha='center', va='bottom', fontweight='bold')

        ax2.set_title("Virtual Array (TX + RX)", color=title_color)
        ax2.set_xlabel("X (Azimuth) [m]", color=text_color)
        ax2.set_ylabel("Y (Elevation) [m]", color=text_color)
        ax2.legend(loc='upper right', facecolor=face_color,
                   edgecolor=grid_color, labelcolor=text_color)
        ax2.grid(color=grid_color, alpha=0.3)
        ax2.tick_params(colors=text_color)
        ax2.axis('equal')

        # Format viền
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("#2a2a4a")

        plt.tight_layout()
        plt.show()


def create_l_shape_mimo(wavelength: float) -> AntennaArray:
    """
    Tạo cấu hình mảng ăng-ten MIMO dạng chữ L phổ biến cho automotive radar.
    Mảng này có 3 TX và 4 RX, tổng hợp ra 12 virtual elements (Azimuth + Elevation).
    Khoảng cách d = lambda / 2.

    Layout vật lý:
      - RX0, RX1, RX2, RX3 xếp dọc theo trục ngang (X).
      - TX0 trùng gốc. TX1 dịch sang ngang để tiếp nối dải RX.
      - TX2 dịch lên trên theo trục dọc (Y) để cung cấp độ phân giải Elevation.

    Args:
        wavelength: Bước sóng của radar lambda [m]
    """
    d = wavelength / 2.0

    # 4 RX đặt trên trục X (Azimuth)
    rx_pos = np.array([
        [0,   0, 0],
        [d,   0, 0],
        [2*d, 0, 0],
        [3*d, 0, 0]
    ])

    # 3 TX đặt dạng chữ L
    tx_pos = np.array([
        [0,   0, 0],   # TX0: Gốc
        [4*d, 0, 0],   # TX1: Dịch ngang 4d để nối tiếp với dãy RX
        [0,   d, 0]    # TX2: Dịch dọc d để tạo trục Elevation
    ])

    return AntennaArray(tx_pos, rx_pos)


def create_sencity_nidar_4x4(wavelength: float) -> AntennaArray:
    """
    Cấu hình mô phỏng mảng ăng-ten 4TX 4RX dựa trên hình ảnh thực tế HUBER+SUHNER SENCITY NiDAR 85211893.

    Phân tích từ Layout thực tế:
      - 4 RX slots được xếp thành hàng ngang ở cạnh dưới.
      - 4 TX slots được xếp thành hàng ngang ở cạnh trên.
      - Vị trí theo trục X (ngang) của 4 TX hoàn toàn trùng khớp/thẳng hàng với 4 RX.

    Kết quả Virtual Array:
      - Khi chập (convolution) 2 mảng ULA 4 phần tử giống hệt nhau, ta KHÔNG có mảng 16 phần tử rời rạc.
      - Thay vào đó, ta thu được mảng ngang 7 phần tử (0d, 1d, 2d, 3d, 4d, 5d, 6d).
      - Các phần tử ở giữa sẽ có sự trùng lặp (Overlap):
        Ví dụ vị trí 3d được tạo ra bởi 4 cặp (TX0+RX3, TX1+RX2, TX2+RX1, TX3+RX0).
      - Tính chất này tạo ra "Spatial Tapering" (Trọng số tam giác: 1-2-3-4-3-2-1),
        giúp búp sóng chính hẹp và triệt tiêu búp sóng phụ (sidelobes) cực tốt cho radar tầm xa.
      - Mảng này chỉ đo được Azimuth, không đo được Elevation.
    """
    d = wavelength / 2.0

    # Giả sử khoảng cách vật lý giữa hàng TX và RX là 50mm (0.05m)
    # Tuy nhiên khoảng cách trục Y này chỉ tạo ra một hằng số pha chung cho mọi phần tử ảo,
    # không giúp tạo ra độ phân giải Elevation.
    H = 0.1

    # 4 RX đặt trên trục X (Azimuth), Y = 0, khoảng cách d
    # Căn giữa mảng RX tại X = 0
    rx_pos = np.array([
        [-1.5 * d, 0, 0],
        [-0.5 * d, 0, 0],
        [0.5 * d, 0, 0],
        [1.5 * d, 0, 0]
    ])

    # 4 TX đặt thẳng hàng với RX, khoảng cách 1.5d
    # Căn giữa mảng TX tại X = 0 (trùng tâm với mảng RX)
    tx_pos = np.array([
        [-2.25 * d, H, 0],   # TX0
        [-0.75 * d, H, 0],   # TX1
        [0.75 * d, H, 0],   # TX2
        [2.25 * d, H, 0]    # TX3
    ])

    return AntennaArray(tx_pos, rx_pos)


def create_uniform_linear_mimo_4x4(wavelength: float) -> AntennaArray:
    """
    Tạo cấu hình mảng MIMO 4TX 4RX dạng ULA (Uniform Linear Array) tiêu chuẩn.

    Layout vật lý:
      - 4 RX xếp cạnh nhau với khoảng cách d = lambda / 2
      - 4 TX xếp cạnh nhau với khoảng cách D = 4 * d

    Kết quả Virtual Array:
      - Sẽ tạo ra chính xác 16 phần tử ăng-ten ảo nằm liên tiếp nhau, khoảng cách đúng d.
      - Loại bỏ hoàn toàn búp sóng phụ (Grating Lobes) bị lỗi.
    """
    d = wavelength / 2.0

    rx_pos = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [2*d, 0, 0],
        [3*d, 0, 0]
    ])

    tx_pos = np.array([
        [0, 0, 0],
        [4*d, 0, 0],
        [8*d, 0, 0],
        [12*d, 0, 0]
    ])

    # Căn giữa mảng ảo vào gốc toạ độ (chỉ để đồ thị đẹp hơn)
    array = AntennaArray(tx_pos, rx_pos)
    center_x = np.mean(array.virtual_pos[:, 0])
    array.tx_pos[:, 0] -= center_x / 2
    array.rx_pos[:, 0] -= center_x / 2
    array.virtual_pos = array._compute_virtual_array()

    return array


if __name__ == "__main__":
    # Test visualization với thông số radar 77 GHz
    # lambda = c / f = 3e8 / 77e9 ~ 3.9 mm
    test_wavelength = 3e8 / 77e9

    # Thử nghiệm cấu hình SENCITY NiDAR 4x4 (4TX, 4RX)
    array = create_uniform_linear_mimo_4x4(test_wavelength)

    print("="*40)
    print("SENCITY NiDAR 4x4 Antenna Array Test")
    print("="*40)
    print(f"Number of TX antennas : {array.n_tx}")
    print(f"Number of RX antennas : {array.n_rx}")
    print(f"Virtual Elements      : {array.n_virtual}")
    print("-"*40)

    array.plot_array_layout()
