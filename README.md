# SVD Image Compressor

A simple PyQt6 application for visualizing image compression using Singular Value Decomposition (SVD).  
Users can load an image, adjust the compression level (rank k), preview results, and download the compressed output.

---

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
