# 🩺 Remote Patient Monitoring System using Federated Learning

## 📌 Overview

This project implements a **privacy-preserving Remote Patient Monitoring (RPM) system** using **Federated Learning (FL)** and wearable IoT health data. The system enables distributed model training across multiple clients (simulated patient devices) without sharing raw data, ensuring **data privacy and security**.

The model predicts the likelihood of **abnormal heart rate conditions** based on physiological indicators such as heart rate, blood oxygen level, activity, sleep, and stress.

---

## 🎯 Objectives

* Implement a **Federated Learning framework** for distributed health data
* Preserve **patient data privacy** by keeping data local to devices
* Detect **abnormal heart rate conditions**
* Evaluate model performance using **precision, recall, and F1-score**
* Simulate a **real-time monitoring and alert system**
* Deploy a **Streamlit web application** for interaction

---

## 🧠 Key Features

* 🔐 Privacy-preserving training (no raw data sharing)
* 📊 Data preprocessing (cleaning, encoding, normalization)
* 🤖 Neural network model for binary classification
* 🔄 Federated Averaging (FedAvg) implementation
* 📉 Performance tracking across training rounds
* 🚨 Alert simulation system
* 🌐 Interactive Streamlit dashboard

---

## 📂 Dataset

* **File:** `unclean_smartwatch_health_data.csv`
* Contains wearable health data including:

  * Heart Rate (BPM)
  * Blood Oxygen Level (%)
  * Step Count
  * Sleep Duration (hours)
  * Stress Level
  * Activity Level

---

## ⚙️ Project Workflow

### 1. Data Preprocessing

* Handle missing values (median imputation)
* Normalize numerical features using `MinMaxScaler`
* Encode categorical variables (One-Hot Encoding)
* Cap outliers to realistic physiological ranges

### 2. Target Variable Engineering

Abnormal heart rate is defined using **data-driven thresholds**:

* High heart rate (top 10%)
* High stress level (top 10%)

This results in a balanced dataset (~18% abnormal cases).

---

### 3. Federated Learning Setup

* Data split into multiple **clients (simulated devices)**
* Each client trains locally
* Only **model weights** are shared with server
* Global model updated using **Federated Averaging (FedAvg)**

---

### 4. Model Architecture

```python
Input (5 features)
→ Dense (64, ReLU)
→ Dense (32, ReLU)
→ Dense (16, ReLU)
→ Dense (1, Sigmoid)
```

---

### 5. Training Configuration

* Communication rounds: **20**
* Clients per round: **20**
* Local epochs: **3**
* Loss: `binary_crossentropy`
* Optimizer: `Adam`

---

### 6. Evaluation Metrics

Final model performance (threshold = **0.6**):

* **Precision:** 0.58
* **Recall:** 0.81
* **F1-score:** 0.67

📌 Interpretation:

* High recall → detects most abnormal cases ✅
* Moderate precision → some false alarms (acceptable in healthcare)

---

### 7. Confusion Matrix

```
[6934 1058]
[ 343 1464]
```

---

### 8. Streamlit Web Application

The project includes an interactive UI for real-time predictions.

#### Features:

* User input via sliders
* Probability prediction
* Risk classification:

  * 🔴 High Risk
  * 🟠 Moderate Risk
  * 🟢 Low Risk
* Clinical interpretation hints

---

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd <project-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

Run the notebook/script to:

* Train federated model
* Save weights (`model_weights.weights.h5`)
* Save scaler (`scaler.pkl`)

---

### 4. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── data/
│   └── unclean_smartwatch_health_data.csv
├── model/
│   ├── model_weights.weights.h5
│   └── scaler.pkl
├── app.py                # Streamlit app
├── requirements.txt
└── README.md
```

---

## 🔍 Key Insights

* Accuracy alone is misleading in healthcare models
* **Recall and F1-score are more critical**
* Threshold tuning significantly improves performance
* Federated Learning is effective for **privacy-sensitive systems**

---

## ⚠️ Limitations

* Synthetic target variable (not clinically validated)
* Some false positives remain
* No real-time device integration yet

---

## 🔮 Future Improvements

* Use real labeled medical datasets
* Add more health indicators (ECG, temperature)
* Implement **SMOTE / oversampling**
* Integrate real-time alerts (SMS, email)
* Deploy as a full-stack web system

---

## 🧾 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is for academic and research purposes.

---



