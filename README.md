#  Fruit Freshness Detection using CNN (EfficientNetB3)

## 🚀 Project Overview

This project is an AI-powered **Fruit Freshness Detection System** that classifies fruits as **Fresh** or **Rotten** using a deep learning model based on **EfficientNetB3**.

The system can also detect **unknown objects** and provide the most likely prediction with confidence scores.


---
## 🌐 Live Demo

👉 https://huggingface.co/spaces/Sojib11111/fruit-freshness-app

---

## 🎯 Features

* ✅ Multi-class classification (10 classes)
* ✅ Fresh vs Rotten detection
* ✅ Unknown object detection using confidence threshold
* ✅ Top-3 predictions with confidence scores
* ✅ User-friendly web interface using Gradio
* ✅ Deployed on Hugging Face Spaces

---

## 🧠 Model Details

* Architecture: EfficientNetB3 (Transfer Learning)

* Input Size: 300x300

* Output Classes:

  * apple_fresh / apple_rotten
  * banana_fresh / banana_rotten
  * mango_fresh / mango_rotten
  * orange_fresh / orange_rotten
  * tomato_fresh / tomato_rotten

* Regularization:

  * Dropout
  * Batch Normalization
  * L2 Regularization

---

## 📊 Prediction Output

* Final Prediction
* Confidence Score (%)
* Top-3 Predictions
* Unknown Object Detection (if confidence < 75%)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* EfficientNetB3
* OpenCV
* NumPy
* Gradio
* PIL

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fruit-freshness-detection.git
cd fruit-freshness-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

---

## 📁 Project Structure

```
├── app.py
├── Fruit_EfficientNetB3_final.h5
├── requirements.txt
└── README.md
```

---

## 🧪 How It Works

1. Upload an image of a fruit
2. Image is resized and preprocessed
3. Model predicts class probabilities
4. Highest probability class is selected
5. If confidence < threshold → marked as "Unknown Object"

---

## 📌 Use Cases

* Smart agriculture
* Food quality inspection
* Retail automation
* Reducing food waste

---

## 📈 Future Improvements

* Grad-CAM visualization for explainability
* Mobile app integration (Flutter)
* Real-time detection using camera
* More fruit categories

---


