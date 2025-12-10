from typing import Optional
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSlider, QCheckBox,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage

from core.image_utils import (
    load_image_as_array,
    get_file_size_mb,
    estimate_jpeg_size_mb,
)
from core.svd_ops import (
    svd_decompose_rgb, reconstruct_rgb,
    mse_rgb, psnr_rgb, energy_for_k, find_k_for_energy,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SVD Image Compression Tool")
        self.resize(1100, 650)

        # 狀態變數
        self.orig_img_arr: Optional[np.ndarray] = None
        self.svd_decomp = None
        self.current_comp_img: Optional[np.ndarray] = None
        self.orig_file_size_mb: Optional[float] = None
        self.current_comp_size_mb: Optional[float] = None

        # 三個預設 K 值
        self.k_light = None   # 95%
        self.k_medium = None  # 97%
        self.k_heavy = None   # 99%

        # ===== layout =====
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # ================= 左邊 =================
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 10, 10)

        # 選圖
        self.btn_select = QPushButton("Select Image")
        self.btn_select.clicked.connect(self.on_select_image)
        header_layout.addWidget(self.btn_select, alignment=Qt.AlignmentFlag.AlignHCenter)

        # 路徑
        self.label_path = QLabel("No file selected")
        self.label_path.setWordWrap(True)
        header_layout.addWidget(self.label_path)

        # 原始大小
        self.label_file_size = QLabel("Original size: N/A")
        header_layout.addWidget(self.label_file_size)

        # 三個 preset 按鈕
        presets_row = QWidget()
        presets_layout = QHBoxLayout(presets_row)
        presets_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_preset_light = QPushButton("Light")
        self.btn_preset_light.setToolTip(
            "Light compression (~95% energy)\n"
            "Good for small uploads when detail isn't critical."
        )
        self.btn_preset_light.clicked.connect(self.on_preset_light)

        self.btn_preset_medium = QPushButton("Medium")
        self.btn_preset_medium.setToolTip(
            "Medium (~97% energy)\n"
            "Balanced option for most documents and daily images."
        )
        self.btn_preset_medium.clicked.connect(self.on_preset_medium)

        self.btn_preset_heavy = QPushButton("Heavy")
        self.btn_preset_heavy.setToolTip(
            "Heavy (~99% energy)\n"
            "High quality for ID photos or fine-detail images."
        )
        self.btn_preset_heavy.clicked.connect(self.on_preset_heavy)

        presets_layout.addWidget(self.btn_preset_light)
        presets_layout.addWidget(self.btn_preset_medium)
        presets_layout.addWidget(self.btn_preset_heavy)
        header_layout.addWidget(presets_row)

        # k Slider
        header_layout.addWidget(QLabel("Compression Rank (k):"))
        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_k.setMinimum(1)
        self.slider_k.setMaximum(100)
        self.slider_k.setValue(50)
        self.slider_k.setEnabled(False)
        self.slider_k.valueChanged.connect(self.on_k_changed)
        header_layout.addWidget(self.slider_k)

        left_layout.addWidget(header, stretch=0)

        # 原圖顯示
        self.label_orig = QLabel("Original")
        self.label_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_orig.setMinimumSize(400, 300)
        self.label_orig.setStyleSheet("border: 1px solid #ccc; background:#f7f7f7;")
        self.label_orig.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        left_layout.addWidget(self.label_orig, stretch=1)

        # ================= 右邊 =================
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)

        comp_header = QWidget()
        comp_header_layout = QVBoxLayout(comp_header)
        comp_header_layout.setContentsMargins(10, 0, 0, 10)

        # Download
        self.btn_download = QPushButton("Download")
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self.on_download_clicked)
        comp_header_layout.addWidget(self.btn_download)

        # Pro mode
        self.chk_pro = QCheckBox("Professional mode")
        self.chk_pro.setChecked(False)
        self.chk_pro.stateChanged.connect(self.on_mode_changed)
        comp_header_layout.addWidget(self.chk_pro)

        # Summary
        self.label_simple = QLabel("Compression: N/A")
        self.label_simple.setStyleSheet("font-weight: bold;")
        comp_header_layout.addWidget(self.label_simple)

        # Pro 指標
        self.label_pro = QLabel("")
        self.label_pro.setVisible(False)
        comp_header_layout.addWidget(self.label_pro)

        right_layout.addWidget(comp_header, stretch=0)

        # 壓縮後圖片
        self.label_comp = QLabel("Compressed")
        self.label_comp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_comp.setMinimumSize(400, 300)
        self.label_comp.setStyleSheet("border: 1px solid #ccc; background:#f7f7f7;")
        self.label_comp.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        right_layout.addWidget(self.label_comp, stretch=1)

        # 加入 root layout
        root_layout.addWidget(left_col, stretch=1)
        root_layout.addWidget(right_col, stretch=1)

    # ================= 選圖片 =================
    def on_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        self.label_path.setText(file_path)
        self.orig_file_size_mb = get_file_size_mb(file_path)
        self.label_file_size.setText(f"Original size: {self.orig_file_size_mb:.2f} MB")

        self.orig_img_arr = load_image_as_array(file_path)
        self.show_image(self.label_orig, self.orig_img_arr)

        # SVD
        self.svd_decomp = svd_decompose_rgb(self.orig_img_arr)
        max_k = int(self.svd_decomp.max_rank)

        # 預設 k
        self.k_light = find_k_for_energy(self.svd_decomp, 0.95)
        self.k_medium = find_k_for_energy(self.svd_decomp, 0.97)
        self.k_heavy = find_k_for_energy(self.svd_decomp, 0.99)

        self.k_light = max(1, min(self.k_light, max_k))
        self.k_medium = max(1, min(self.k_medium, max_k))
        self.k_heavy = max(1, min(self.k_heavy, max_k))

        # Slider
        self.slider_k.setMaximum(max_k)
        self.slider_k.setValue(self.k_medium)
        self.slider_k.setEnabled(True)

        # 預設更新
        self.update_compression_view(self.k_medium)

    # ================= numpy → QLabel =================
    def show_image(self, target_label: QLabel, img_arr: np.ndarray):
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        h, w, ch = img_arr.shape
        bytes_per_line = ch * w

        qimg = QImage(img_arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        available = target_label.size()
        scaled = pix.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        target_label.setPixmap(scaled)

    # ================= 更新畫面 =================
    def update_compression_view(self, k: int):
        if self.svd_decomp is None:
            return

        k = int(k)
        comp_img = reconstruct_rgb(self.svd_decomp, k)
        self.current_comp_img = comp_img
        self.show_image(self.label_comp, comp_img)

        energy = energy_for_k(self.svd_decomp, k)
        mse = mse_rgb(self.orig_img_arr, comp_img)
        psnr = psnr_rgb(self.orig_img_arr, comp_img)
        rank_perc = 100 * k / float(self.svd_decomp.max_rank)

        # JPEG 真實估計
        est_size = estimate_jpeg_size_mb(comp_img, quality=90)
        ratio = est_size / self.orig_file_size_mb

        self.label_simple.setText(
            f"Rank k = {k} · {rank_perc:.1f}% of full rank\n"
            f"Estimated JPEG size: {est_size:.2f} MB ({ratio*100:.1f}% of original)"
        )

        self.label_pro.setText(
            f"Energy kept: {energy*100:.2f}%\n"
            f"PSNR: {psnr:.2f} dB\n"
            f"MSE : {mse:.2f}"
        )

        self.btn_download.setEnabled(True)

    # ================= Slider 事件 =================
    def on_k_changed(self, value: int):
        """When the slider moves, update the compressed preview."""
        self.update_compression_view(int(value))

    # ================= preset =================
    def on_preset_light(self):
        if self.k_light:
            self.slider_k.setValue(self.k_light)

    def on_preset_medium(self):
        if self.k_medium:
            self.slider_k.setValue(self.k_medium)

    def on_preset_heavy(self):
        if self.k_heavy:
            self.slider_k.setValue(self.k_heavy)

    # ================= 模式切換 =================
    def on_mode_changed(self, state):
        self.label_pro.setVisible(self.chk_pro.isChecked())

    # ================= 視窗縮放 =================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.orig_img_arr is not None:
            self.show_image(self.label_orig, self.orig_img_arr)
        if self.current_comp_img is not None:
            self.show_image(self.label_comp, self.current_comp_img)

    # ================= 下載 =================
    def on_download_clicked(self):
        if self.current_comp_img is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Image",
            "compressed.jpg",
            "JPEG Image (*.jpg);;PNG Image (*.png)"
        )
        if not save_path:
            return

        from PIL import Image
        arr = np.clip(self.current_comp_img, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(save_path, quality=90)