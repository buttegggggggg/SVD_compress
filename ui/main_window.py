from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSlider, QCheckBox,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage
import numpy as np

from core.image_utils import (
    load_image_as_array,
    get_file_size_mb,
    estimate_compressed_size_mb,
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
        self.orig_img_arr = None
        self.svd_decomp = None
        self.current_comp_img = None
        self.orig_file_size_mb = None
        self.current_comp_size_mb = None
        self.k_light = None     # ~90% energy
        self.k_medium = None    # ~95% energy
        self.k_heavy = None     # ~99% energy

        # ===== 中央容器 =====
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout()
        central.setLayout(root_layout)

        # ===================== 左邊：Original 欄 =====================
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)

        # --- Original header（選圖 + 路徑 + 原始大小 + preset + k slider）---
        header_orig = QWidget()
        header_orig_layout = QVBoxLayout(header_orig)
        header_orig_layout.setContentsMargins(0, 0, 10, 10)

        # Select Image 按鈕
        self.btn_select = QPushButton("Select Image")
        self.btn_select.clicked.connect(self.on_select_image)
        header_orig_layout.addWidget(
            self.btn_select,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        # 檔案路徑
        self.label_path = QLabel("No file selected")
        self.label_path.setWordWrap(True)
        header_orig_layout.addWidget(self.label_path)

        # 原始檔案大小
        self.label_file_size = QLabel("Original size: N/A")
        header_orig_layout.addWidget(self.label_file_size)

        # === 建議壓縮 presets（Light / Medium / Heavy）===
        presets_row = QWidget()
        presets_layout = QHBoxLayout(presets_row)
        presets_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_preset_light = QPushButton("Light")
        self.btn_preset_light.setToolTip(
            "Light compression (~95% energy)\n"
            "Best for: scanned documents, tables, receipts\n"
            "Note: slight blurring is acceptable for text-only content."
        )

        self.btn_preset_medium = QPushButton("Medium")
        self.btn_preset_medium.setToolTip(
            "Medium compression (~97% energy)\n"
            "Best for: ID photos, profile pictures, documents with images\n"
            "Balanced quality and size; ideal for uploading forms."
        )

        self.btn_preset_heavy = QPushButton("Heavy")
        self.btn_preset_heavy.setToolTip(
            "Heavy compression (~99% energy)\n"
            "Best for: color photos, scenery images, high-resolution pictures\n"
            "Preserves fine details with minimal quality loss."
        )

        # 綁定三個 preset 按鈕事件
        self.btn_preset_light.clicked.connect(self.on_preset_light)
        self.btn_preset_medium.clicked.connect(self.on_preset_medium)
        self.btn_preset_heavy.clicked.connect(self.on_preset_heavy)

        presets_layout.addWidget(self.btn_preset_light)
        presets_layout.addWidget(self.btn_preset_medium)
        presets_layout.addWidget(self.btn_preset_heavy)

        header_orig_layout.addWidget(presets_row)

        # === Compression Rank (k) Slider ===
        header_orig_layout.addWidget(QLabel("Compression Rank (k):"))
        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_k.setMinimum(1)
        self.slider_k.setMaximum(100)
        self.slider_k.setValue(50)
        self.slider_k.setEnabled(False)
        self.slider_k.valueChanged.connect(self.on_k_changed)
        header_orig_layout.addWidget(self.slider_k)

        # 把 header 放進左欄
        left_layout.addWidget(header_orig, stretch=0)

        # --- Original 圖片 ---
        self.label_orig = QLabel("Original")
        self.label_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_orig.setStyleSheet("border: 1px solid #ccc; background: #f7f7f7;")
        self.label_orig.setMinimumSize(400, 300)
        self.label_orig.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        left_layout.addWidget(self.label_orig, stretch=1)

        # ===================== 右邊：Compressed 欄 =====================
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)

        # --- Compressed header（download + mode + 數值）---
        header_comp = QWidget()
        header_comp_layout = QVBoxLayout(header_comp)
        header_comp_layout.setContentsMargins(10, 0, 0, 10)

        # Download 按鈕
        self.btn_download = QPushButton("Download")
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self.on_download_clicked)
        header_comp_layout.addWidget(self.btn_download)

        # Pro mode checkbox（只控制顯示多寡）
        self.chk_pro = QCheckBox("Professional mode")
        self.chk_pro.setChecked(False)
        self.chk_pro.stateChanged.connect(self.on_mode_changed)
        header_comp_layout.addWidget(self.chk_pro)

        # 簡單模式文字（壓縮 summary）
        self.label_simple = QLabel("Compression: N/A")
        self.label_simple.setStyleSheet("font-weight: bold;")
        header_comp_layout.addWidget(self.label_simple)

        # 專業模式文字（Energy / PSNR / MSE）
        self.label_pro = QLabel("")
        self.label_pro.setVisible(False)
        header_comp_layout.addWidget(self.label_pro)

        right_layout.addWidget(header_comp, stretch=0)

        # --- Compressed 圖片 ---
        self.label_comp = QLabel("Compressed")
        self.label_comp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_comp.setStyleSheet("border: 1px solid #ccc; background: #f7f7f7;")
        self.label_comp.setMinimumSize(400, 300)
        self.label_comp.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        right_layout.addWidget(self.label_comp, stretch=1)

        # ===== 放進 root layout =====
        root_layout.addWidget(left_col, stretch=1)
        root_layout.addWidget(right_col, stretch=1)

    # ===================== 選圖片 =====================
    def on_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        # 路徑
        self.label_path.setText(file_path)

        # 原始檔案大小（MB）
        self.orig_file_size_mb = get_file_size_mb(file_path)
        if self.orig_file_size_mb is not None:
            self.label_file_size.setText(
                f"Original size: {self.orig_file_size_mb:.2f} MB"
            )
        else:
            self.label_file_size.setText("Original size: N/A")

        # 讀圖 + 顯示原圖
        self.orig_img_arr = load_image_as_array(file_path)
        self.show_image(self.label_orig, self.orig_img_arr)

        # 做 full SVD
        self.svd_decomp = svd_decompose_rgb(self.orig_img_arr)

        # --- 計算三種 energy 對應的 k ---
        max_k = int(self.svd_decomp.max_rank)

        self.k_light = find_k_for_energy(self.svd_decomp, 0.95)
        self.k_medium = find_k_for_energy(self.svd_decomp, 0.97)
        self.k_heavy = find_k_for_energy(self.svd_decomp, 0.99)

        # 安全界線
        self.k_light = max(1, min(self.k_light, max_k))
        self.k_medium = max(1, min(self.k_medium, max_k))
        self.k_heavy = max(1, min(self.k_heavy, max_k))

        # slider 範圍 & 預設值 → 用「中度壓縮」
        self.slider_k.setMaximum(max_k)
        default_k = self.k_medium if self.k_medium is not None else min(50, max_k)
        self.slider_k.setValue(default_k)
        self.slider_k.setEnabled(True)

        # 顯示預設壓縮
        self.update_compression_view(default_k)

    # ===================== 顯示 numpy → Label =====================
    def show_image(self, target_label: QLabel, img_arr: np.ndarray):
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        h, w, ch = img_arr.shape
        bytes_per_line = ch * w

        qimg = QImage(
            img_arr.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg)

        # 以目前 label 大小等比例縮放
        available_size = target_label.size()
        if available_size.width() <= 0 or available_size.height() <= 0:
            available_size = QSize(400, 300)

        scaled = pix.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        target_label.setPixmap(scaled)

    # ===================== 更新壓縮圖片 + 顯示資訊 =====================
    def update_compression_view(self, k: int):
        if self.svd_decomp is None or self.orig_img_arr is None:
            return

        k = int(k)

        # 1. 重建壓縮圖並顯示
        comp_img = reconstruct_rgb(self.svd_decomp, k)
        self.current_comp_img = comp_img
        self.show_image(self.label_comp, comp_img)

        # 2. 指標計算
        energy = energy_for_k(self.svd_decomp, k)          # 0~1
        mse = mse_rgb(self.orig_img_arr, comp_img)
        psnr = psnr_rgb(self.orig_img_arr, comp_img)
        rank_perc = 100.0 * k / float(self.svd_decomp.max_rank)

        # 估算壓縮後檔案大小
        size_mb = None
        size_ratio = None
        if self.orig_file_size_mb is not None:
            size_mb = estimate_compressed_size_mb(
                self.orig_file_size_mb,
                k,
                int(self.svd_decomp.max_rank),
            )
            self.current_comp_size_mb = size_mb
            size_ratio = (
                size_mb / self.orig_file_size_mb
                if self.orig_file_size_mb > 0 else None
            )

        # 3. 簡單模式 summary
        line1 = f"Rank k = {k}   ·   {rank_perc:.1f}% of full rank"
        if size_mb is not None and size_ratio is not None:
            line2 = (
                f"Estimated size: {size_mb:.2f} MB   ·   "
                f"{size_ratio*100:.1f}% of original"
            )
        else:
            line2 = "Estimated size: N/A"

        self.label_simple.setText(line1 + "\n" + line2)

        # 4. 專業模式
        self.label_pro.setText(
            "Energy kept: {:.2f}%\n"
            "PSNR: {:.2f} dB\n"
            "MSE : {:.2f}".format(energy * 100.0, psnr, mse)
        )

        # 5. 啟用 Download 按鈕
        self.btn_download.setEnabled(True)

    # ===================== Slider 事件 =====================
    def on_k_changed(self, value: int):
        self.update_compression_view(value)

    def on_preset_light(self):
        """輕度壓縮：~90% energy。"""
        if self.svd_decomp is None or self.k_light is None:
            return
        k = int(self.k_light)
        self.slider_k.setValue(k)   # 會觸發 on_k_changed

    def on_preset_medium(self):
        """中度壓縮：~95% energy。"""
        if self.svd_decomp is None or self.k_medium is None:
            return
        k = int(self.k_medium)
        self.slider_k.setValue(k)

    def on_preset_heavy(self):
        """重度壓縮（高品質）：~99% energy。"""
        if self.svd_decomp is None or self.k_heavy is None:
            return
        k = int(self.k_heavy)
        self.slider_k.setValue(k)

    # ===================== 模式切換 =====================
    def on_mode_changed(self, state: int):
        self.label_pro.setVisible(self.chk_pro.isChecked())

    # ===================== 視窗縮放：重畫圖片 =====================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.orig_img_arr is not None:
            self.show_image(self.label_orig, self.orig_img_arr)
        if self.current_comp_img is not None:
            self.show_image(self.label_comp, self.current_comp_img)

    # ===================== 下載壓縮後圖片 =====================
    def on_download_clicked(self):
        if self.current_comp_img is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Image",
            "compressed_image.jpg",
            "JPEG Image (*.jpg);;PNG Image (*.png)"
        )
        if not save_path:
            return

        from PIL import Image
        img_to_save = np.clip(self.current_comp_img, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_to_save)

        try:
            pil_img.save(save_path)
            print(f"Saved compressed image to {save_path}")
        except Exception as e:
            print("Save failed:", e)