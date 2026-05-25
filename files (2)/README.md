# 🧠 Image Classification Model Using CNN

**Student:** Kesavasakthi &nbsp;|&nbsp; **Roll No:** RA2311031010001  
**Degree:** B.Tech Computer Science Engineering  
**Dataset:** CIFAR-10 (60,000 images · 10 classes)  
**Framework:** TensorFlow / Keras · Flask · AWS

---

## 📁 Project Structure

```
ImageClassification-CNN/
│
├── model/
│   ├── train_model.py       ← CNN training script (run this first!)
│   ├── cnn_cifar10.h5       ← Saved model (generated after training)
│   ├── metrics.json         ← Test accuracy & stats (auto-generated)
│   └── plots/               ← Training curves, confusion matrix, predictions
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       └── sample_predictions.png
│
├── backend/
│   └── app.py               ← Flask REST API server
│
├── frontend/
│   └── index.html           ← Blue-themed web UI
│
├── notebooks/
│   └── CNN_CIFAR10.ipynb    ← Jupyter notebook (step-by-step)
│
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the CNN Model
```bash
python model/train_model.py
```
This will:
- Download CIFAR-10 automatically (~163 MB, one-time)
- Train the CNN for 30 epochs (~10–20 min on CPU, ~3 min on GPU)
- Save `model/cnn_cifar10.h5`
- Generate plots in `model/plots/`

### Step 3 — Start the Flask API & Open Frontend
```bash
python backend/app.py
```
Then open `frontend/index.html` in your browser.  
The API runs at **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/health` | Check API & model status |
| GET | `/classes` | List all 10 CIFAR-10 classes |
| POST | `/classify` | Classify uploaded image (form-data: `image`) |
| POST | `/classify_base64` | Classify base64 JSON image |

### Example Request
```bash
curl -X POST http://localhost:5000/classify \
     -F "image=@cat.jpg"
```

### Example Response
```json
{
  "success": true,
  "top_class": "Cat",
  "top_emoji": "🐱",
  "top_confidence": 73.45,
  "predictions": [
    {"rank": 1, "class": "Cat",  "emoji": "🐱", "confidence": 73.45},
    {"rank": 2, "class": "Dog",  "emoji": "🐶", "confidence": 12.30},
    {"rank": 3, "class": "Deer", "emoji": "🦌", "confidence": 7.12},
    {"rank": 4, "class": "Frog", "emoji": "🐸", "confidence": 4.01},
    {"rank": 5, "class": "Bird", "emoji": "🐦", "confidence": 3.12}
  ],
  "inference_ms": 28.4,
  "model": "CNN — CIFAR-10 (TensorFlow/Keras)"
}
```

---

## 🧠 CNN Architecture

```
Input (32×32×3)
    ↓
Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool(2×2) → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool(2×2) → Dropout(0.25)
    ↓
Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool(2×2) → Dropout(0.25)
    ↓
Flatten → Dense(512) → BN → Dropout(0.5)
    ↓
Dense(10) → Softmax → Prediction
```

| Parameter | Value |
|-----------|-------|
| Input Size | 32 × 32 × 3 |
| Total Params | ~1.2 Million |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Test Accuracy | ~70% |

---

## 🗂 CIFAR-10 Classes (10)

| # | Class | Emoji |
|---|-------|-------|
| 0 | Airplane | ✈️ |
| 1 | Automobile | 🚗 |
| 2 | Bird | 🐦 |
| 3 | Cat | 🐱 |
| 4 | Deer | 🦌 |
| 5 | Dog | 🐶 |
| 6 | Frog | 🐸 |
| 7 | Horse | 🐴 |
| 8 | Ship | 🚢 |
| 9 | Truck | 🚛 |

---

## ☁️ AWS Deployment

| Service | Purpose |
|---------|---------|
| **SageMaker** | Train CNN on GPU, deploy real-time endpoint |
| **S3** | Store images & model artifacts (.h5) |
| **EC2** | Host Flask REST API server |
| **Lambda** | Serverless image preprocessing (resize/normalize) |

**Request Flow:**  
`User → EC2 (Flask) → Lambda (preprocess) → SageMaker (CNN) → S3 (store) → JSON Response`

---

## 📊 Expected Results

- **Training Accuracy:** ~78%
- **Validation Accuracy:** ~70%
- **Best Classes:** Automobile (82%), Ship (81%)
- **Hardest Classes:** Cat (55%), Dog (60%) — visually similar

---

## 📓 Jupyter Notebook

For a step-by-step interactive walkthrough:
```bash
pip install jupyter
jupyter notebook notebooks/CNN_CIFAR10.ipynb
```
