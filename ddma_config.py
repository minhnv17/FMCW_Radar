"""
ddma_config.py — Cấu hình kỹ thuật Doppler Division Multiple Access (DDMA)

Module này chịu trách nhiệm tạo ra ma trận lệch pha (phase shift matrix)
để áp dụng cho các ăng-ten phát (TX) khi chúng phát đồng thời.
Mỗi TX sẽ có một tốc độ dịch pha khác nhau qua từng chirp,
làm cho phổ Doppler của chúng bị phân tách vào các khoảng riêng biệt.
"""

import numpy as np


class DDMAConfig:
    @staticmethod
    def get_phase_matrix(n_tx: int, n_chirps: int, mode: str = "standard") -> np.ndarray:
        """
        Tạo ma trận độ lệch pha cho các ăng-ten TX qua từng chirp.

        Args:
            n_tx: Số lượng ăng-ten phát.
            n_chirps: Số lượng chirp trong một frame.
            mode: Chế độ mã hóa pha.
                  - "standard": \\Delta\\phi_{t,m} = t * (2\\pi / n_tx) * m
                  (Dịch phổ Doppler thành n_tx dải bằng nhau)

        Returns:
            phase_matrix: Ma trận numpy kích thước (n_tx, n_chirps)
                          chứa giá trị pha (radian).
        """
        phase_matrix = np.zeros((n_tx, n_chirps), dtype=np.float64)

        if mode == "standard":
            # Mã hóa pha tuyến tính: \Delta\phi_{t,m} = t * (2\pi / n_tx) * m
            for t in range(n_tx):
                phase_step = t * (2 * np.pi / n_tx)
                phase_matrix[t, :] = (phase_step * np.arange(n_chirps)) % (2 * np.pi)
        else:
            raise ValueError(f"Unknown DDMA mode: {mode}")

        return phase_matrix


if __name__ == "__main__":
    # Test
    pm = DDMAConfig.get_phase_matrix(4, 10)
    print(np.degrees(pm))
