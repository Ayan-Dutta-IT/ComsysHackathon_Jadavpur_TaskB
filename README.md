# 🔍 Face Matching with Distorted Inputs

A robust face verification system using a Siamese Neural Network. The goal is to determine whether a distorted face image belongs to a known identity in the reference set.

---

## 📁 Project Structure

```
FaceMatchingProject/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│       └── person_id/
│           ├── clean1.jpg
│           └── distorted/
│               ├── dist1.jpg
│               └── dist2.jpg
├── model/
│   └── siamese_model.h5
├── train.py
├── test.py
├── utils.py
└── README.md
```

Each person's folder contains one or more **clean images** and a `distortion/` subfolder with distorted versions of those images.

---

## 🧠 Model Overview

- **Architecture**: Siamese Network
- **Backbone**: MobileNetV2 (pretrained, frozen)
- **Embedding Layer**: Dense(128)
- **Comparison**: Absolute difference of embeddings
- **Head**: Dense(1, activation='sigmoid')
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam

---

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Structure

Organize the dataset under `data/train/`, `data/val/`, and `data/test/` following the structure shown above.

---

## 🚀 Training( directly test with saved model weights under model folder )

```bash
python train.py
```

- Auto-generates positive and negative pairs.
- Enforces 1:1 balance between positive and negative samples.
- Uses validation data for monitoring.
- Progress bars included for preprocessing.
- Saves model to `model/siamese_model.h5`.

---

## 🧪 Testing
The model folder should be created and weights to be downloaded from below drive link,
Link - https://drive.google.com/file/d/1LoOoB0grg2rIVI2RJETyKy-ljvq_nbdJ/view?usp=sharing

```bash
python test.py
```

- Evaluates the model using the `data/test/` folder.
- Reports:
  - ✅ Top-1 Accuracy
  - 📊 Macro-averaged F1 Score

---


## 📊 Metrics

- **Top-1 Accuracy**: Measures how often the highest-scoring match is correct.
- **Macro F1-Score**: Ensures balanced performance across all identities.

---

## ✨ Features

- 🔄 Balanced pair sampling
- 🖼️ Handles multiple clean/distorted images per person
- 🔄 Modular and extendable codebase
- 💾 Saves and loads trained model
- ✅ Handles missing/corrupt images gracefully

---

## 📦 Dependencies

- `tensorflow`
- `scikit-learn`
- `opencv-python`
- `numpy`
- `matplotlib`
- `tqdm`
