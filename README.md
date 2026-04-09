---
title: Bilingual Hybrid CRNN Handwriting OCR
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 Bilingual Hybrid CRNN Handwriting OCR

A comprehensive deep learning system for recognizing handwritten text in both **English** and **Hindi**. This project features a hybrid architecture combining a custom-trained **CRNN (CNN + BiLSTM + CTC)** with the **EasyOCR** engine, accessible via a user-friendly dark-mode web dashboard.

---

## 📸 Key Features

- 🌍 **Bilingual Support:** Recognition for English (alphanumeric) and Hindi (Devanagari script).
- 🧠 **Engine Options:** 
  - **English (CRNN):** Custom model optimized for character-wise English handwriting trained on EMNIST.
  - **Hindi/Multi (EasyOCR):** Integrated engine handles complex Hindi matras and connected scripts.
- 🎨 **Premium Dashboard:** Real-time canvas drawing, batch history with thumbnails, and engine switching.
- ⚡ **Advanced Preprocessing:** 
  - **Auto-Crop:** Removes white margins to focus on the text. 
  - **Intelligent Inversion:** Detects background type (paper vs. digital) and adjusts for the model automatically.
  - **Normalization:** Grayscale conversion and contrast enhancement (CLAHE).

---

## 🏗️ Architecture

```
Input Image (1 × 32 × 32)
        ↓
CNN Backbone (7 conv blocks)    ← extracts visual features
        ↓
Reshape to sequence             ← width becomes time steps
        ↓
BiLSTM × 2 layers               ← reads sequence left ↔ right
        ↓
CTC Decode                      ← collapses to final text
        ↓
Recognized Text
```

### Detailed Component Specs
| Component | Detail |
|---|---|
| **CNN Backbone** | 7 conv blocks, 1 → 64 → 128 → 256 → 512 channels |
| **RNN Layers** | 2× Bidirectional LSTM, hidden size 256 |
| **Loss Function** | CTC Loss (alignment-free) |
| **Input size** | 1 × 32 × 32 grayscale (normalized) |
| **Vocabulary** | 63 Classes (0-9, A-Z, a-z + CTC blank) |

---

## 🔍 How It Works

### 1. CNN — The Eyes
The 7-layer convolutional backbone progressively extracts features from the input image. Early layers detect edges and strokes, later layers detect full character shapes. MaxPool layers with asymmetric kernels `(2,1)` collapse height while preserving width — keeping character position information intact.

### 2. BiLSTM — The Memory
The feature map is reshaped so each column becomes a time step. Two bidirectional LSTM layers read this sequence left→right and right→left simultaneously, giving every position context from both directions.

### 3. CTC — The Decoder
Connectionist Temporal Classification collapses repeated predictions and blanks into clean text without needing manual character segmentation. `HHHH--ii--` becomes `Hi`.

### 4. Hybrid Engine Logic
The server intelligently routes requests based on user selection:
- **English Mode:** Uses character-wise slicing and the EMNIST-trained CRNN Baseline.
- **Hindi/Multi Mode:** Uses EasyOCR with its massive language dictionary for word-perfect results including Matras.

---

## 📁 Project Structure

```bash
crnn-handwriting-ocr/
│
├── server.py                 # Hybrid Flask API (Primary Entry Point)
├── model.py                  # CRNN model architecture definition
├── dataset.py                # Vocabulary mapping (63 ASCII classes)
├── dashboard.html            # Premium Web UI
│
├── checkpoints/              # Model weights
│   └── best_crnn.pth         # English Baseline (88% acc)
│
├── results/                  # Performance charts and analysis
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── predictions.png
│
├── venv/                     # Python environment
└── requirements.txt          # Dependencies
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Baseline (Optional)
Open the original `CRNN_Handwriting_Recognition.ipynb` in Jupyter to see how the English model was trained on the EMNIST dataset.

### 3. Start the Server
```bash
python server.py
```
*Note: On first run, it will download ~100MB of EasyOCR weights automatically.*

### 4. Open the Dashboard
Open `dashboard.html` in any browser.
- **English:** Select "English (CRNN)" for separated characters.
- **Hindi:** Select "Hindi/Multi (EasyOCR)" for full words with matras.

---

## � API Reference

### `POST /predict`
**Request Body:**
```json
{
  "image": "<base64_string>",
  "engine": "crnn" or "easyocr"
}
```

**Response Body:**
```json
{
  "recognized_text": "Akhil",
  "confidence": 0.97,
  "mode": "CRNN - Sliced",
  "model_accuracy": 88.0
}
```

---

Check out the live demo here: [khan0107/crnn-handwritten-character-recognition-hindi-english-ocr](https://huggingface.co/spaces/khan0107/crnn-handwritten-character-recognition-hindi-english-ocr)

---

## 🚀 Deployment (Hugging Face Spaces)

This project is fully containerized and compatible with Hugging Face Spaces.

1. **Dockerized Setup:** Uses `python:3.9-slim` with optimized OpenCV dependencies.
2. **Auto-Port Detection:** Backend automatically detects `PORT` for cloud environments.
3. **Git LFS:** Uses Git Large File Storage for model weights (`*.pth`).

To deploy your own:
1. Create a "Docker" Space on Hugging Face.
2. Push this repository to the Space remote.
3. Hugging Face will automatically build and launch the dashboard.

---

## 📊 Baseline Training Results (English)

| Epoch | Accuracy |
|---|---|
| 1 | 78.23% |
| 8 | 85.30% |
| 17 | **88.0%** ← best |

- **Dataset:** EMNIST ByClass (60,000+ samples)
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing

---

## 🔍 Preprocessing Pipeline

To ensure maximum accuracy, every input goes through:
1. **Binarization:** Converting color/gray to sharp black & white via Otsu's Thresholding.
2. **Auto-Cropping:** Using `findNonZero` to identify the bounding box of the script.
3. **Resizing:** Scaling to 32px height while maintaining aspect ratio (crucial for CRNN).
4. **Padding:** Adding consistent margins to prevent edge-cutting.

---

## 🤝 Project Context
This project was developed as a comprehensive Deep Learning showcase, demonstrating model training, computer vision preprocessing, and professional full-stack integration. 

---

## 📄 License
MIT License
