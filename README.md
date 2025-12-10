# SVD Image Compressor

A simple PyQt6 application for visualizing image compression using Singular Value Decomposition (SVD).  
Users can load an image, adjust the compression level (rank k), preview results, and download the compressed output.

---
##  Version 1.0 — First Official Release

After several iterations and internal testing, this is the **first official stable version** of the SVD Image Compression Tool.

###  Key Improvements in v1.0
- **Replaced approximate size estimation with JPEG-based prediction**  
  The previous linear model has been removed. The app now uses an actual in-memory JPEG encode to estimate the compressed file size, resulting in **far more accurate size predictions**.

- **Preset compression levels updated**  
  Energy thresholds revised to **95% / 97% / 99%** for more practical real-world usage.

- **UI refinement**  
  Improved layout for presets, professional mode, and compressed-image information display.

--
## 1. Create a Virtual Environment

Inside the project folder, run:

```bash
python3 -m venv venv
```

Activate the environment:

### macOS / Linux
```bash
source venv/bin/activate
```

### Windows
```bash
venv\Scripts\activate
```

---

## 2. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 3. Run the Application

```bash
python main.py
```

---

## Folder Structure
```
SVD_GUI_APP/
│── core/
│   ├── image_utils.py
│   ├── svd_ops.py
│── ui/
│   ├── main_window.py
│── main.py
│── requirements.txt
```

---

## Notes
- This application uses PyQt6 for the GUI.
- SVD-based compression allows users to manually control image quality via the rank k.
- Works on macOS, Windows, and Linux as long as PyQt6 is installed.
