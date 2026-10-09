# 💊 Smart-Cure — Medicine Recommendation System

An AI-powered web application that uses a **Random Forest ML model** trained on the **Kaggle Disease-Symptom Dataset** to predict diseases from symptoms and provide medicine, diet, and lifestyle recommendations.

> Built by **Vaishnavi** (22AI059) · Dept of AI & ML · SSIT, Tumakuru · 2024–25

---

## 🌐 Live Demo
👉 `https://tech-master22.github.io/medicine-recommendation-system`

---

## ✨ Features
| Feature | Description |
|---|---|
| 🤖 ML Model | Random Forest trained on Kaggle dataset — **100% accuracy** |
| 🗃 Dataset | 41 diseases · 131 symptoms · 4920 training samples |
| 🩺 Symptom Checker | Search + select symptoms with autocomplete |
| 📊 Confidence Score | Model prediction confidence % shown |
| 🏆 Top 3 Predictions | Top 3 possible diseases with probability bars |
| 💊 Medicines | Curated medicines with dosage + Apollo order link |
| 🥗 Diet Plans | Disease-specific nutrition advice |
| 🏃 Workout Guide | Condition-specific activity recommendations |
| 🎤 Voice Input | Speech-to-text symptom entry |
| 🚨 Emergency | Ambulance, WhatsApp doctor, Google Maps |
| 🏥 Live Hospitals | GPS-based real-time nearby hospital map |

---

## 🗂 Project Structure
```
smart-cure/
├── app.py                    ← Flask backend + ML prediction
├── requirements.txt          ← Python dependencies
├── templates/
│   └── index.html            ← Main HTML (Jinja2)
├── static/
│   └── css/style.css         ← All styles
├── model/
│   ├── rf_model.pkl          ← Trained Random Forest model
│   ├── symptoms_list.pkl     ← 131 symptoms feature list
│   └── classes.pkl           ← 41 disease class labels
└── data/
    ├── dataset.csv           ← Kaggle: symptom-disease mapping
    ├── symptom_Description.csv ← Disease descriptions
    ├── symptom_precaution.csv  ← Precautions per disease
    ├── Symptom-severity.csv    ← Symptom severity weights
    ├── description.json      ← Processed descriptions
    ├── precaution.json        ← Processed precautions
    └── severity.json          ← Processed severity weights
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Tech-master22/medicine-recommendation-system.git
cd medicine-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Flask server
python app.py

# 4. Open browser
# Go to: http://localhost:5000
```

---

## 🤖 ML Model Details
| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | Kaggle Disease-Symptom Dataset |
| Training samples | 4920 |
| Diseases | 41 |
| Symptoms (features) | 131 |
| Train/Test split | 80/20 |
| Accuracy | **100%** |

---

## 🛠 Tech Stack
- **Backend:** Python 3, Flask
- **ML:** scikit-learn (Random Forest, Decision Tree)
- **Data:** pandas, numpy
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **APIs:** Web Speech API, Geolocation API, Google Maps

---

## ⚠️ Disclaimer
This system provides **preliminary health guidance only** and is **not a substitute for professional medical advice**. Always consult a qualified doctor.

---

## 📄 Academic Info
Mini Project · VI Semester BE (AI & ML) · SSIT Tumakuru · 2024–25  
Guide: **Dr. Manjula Yerva**, Associate Professor, Dept of AI&ML
