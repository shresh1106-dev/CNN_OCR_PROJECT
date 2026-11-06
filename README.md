# Industry 4.0: Autonomous Asset Tracking via Multi-Digit OCR & CNN

An end-to-end computer vision and deep learning solution simulating Autonomous Mobile Robots (AMRs) and Automated Guided Vehicles (AGVs) reading multi-digit container codes, segmenting characters, predicting serial numbers using a lightweight CNN trained on MNIST, and automatically updating ERP/WMS inventory logs.

---

## 📌 Features

- **MNIST-Trained CNN**: Compact CNN architecture (~56k parameters) optimized for edge deployment.
- **OpenCV Character Segmentation**: Bounding-box thresholding and contour sorting for multi-digit tag extraction.
- **Automated ERP Logging**: Outputs timestamped inventory records in standard JSON format.
- **Interactive Streamlit Web Dashboard**: GUI to simulate AMR scans and view real-time predictions.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/your-username/industry4.0-ocr-asset-tracking.git](https://github.com/your-username/industry4.0-ocr-asset-tracking.git)
cd industry4.0-ocr-asset-tracking
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt