# AegisAI – Explainable Disaster Severity Prediction

AegisAI is an end-to-end explainable AI system that predicts disaster severity
(Low / Medium / High) using historical FEMA disaster data.  
The system provides confidence-aware predictions and human-readable decision
briefings through an interactive web application.

---

## 🚀 Features
- Machine Learning model trained on real FEMA disaster declarations
- Predicts disaster severity: **Low / Medium / High**
- Confidence-aware predictions with probability breakdown
- Explainable AI briefings for decision support
- Interactive Streamlit web application
- Fully runnable on a local laptop (no cloud required)

---

## 🧠 How It Works
1. Historical FEMA disaster data is cleaned and processed
2. Temporal and categorical features are engineered
3. A Random Forest classifier is trained with class balancing
4. The model outputs severity predictions with confidence scores
5. Results are displayed in a user-friendly web interface

---

## 🛠 Tech Stack
- **Python**
- **pandas, NumPy**
- **scikit-learn**
- **Streamlit**
- **FEMA Disaster Declarations Dataset**

---

## 📊 Model Highlights
- ~90% overall accuracy
- Strong recall for high-severity disasters
- Confidence-aware predictions
- Interpretable feature contributions

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
