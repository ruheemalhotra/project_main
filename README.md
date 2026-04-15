# Blood Cell Anomaly Detection using Autoencoder

## 1. Project Overview

This project implements an unsupervised anomaly detection system using an autoencoder neural network.
The model learns normal patterns in blood cell data and identifies anomalies based on reconstruction error.

---

## 2. Features

* Handles both numerical and categorical data
* Uses one-hot encoding for categorical variables
* Detects anomalies using reconstruction error
* Saves results in structured CSV files
* Automated execution using GitHub Actions
* Docker support for future deployment

---

## 3. Project Structure

```
project_main/
│── data/
│   └── blood_cell_anomaly_detection.csv
│── src/
│   └── train.py
│── outputs/
│   ├── anomaly_results.csv
│   ├── top_anomalies.csv
│   └── reconstructed_data.csv
│── model/
│   ├── autoencoder.keras
│   ├── scaler.pkl
│   └── threshold.pkl
│── .github/workflows/
│   └── main.yml
│── Dockerfile
│── .dockerignore
│── requirements.txt
│── README.md
```

---

## 4. Methodology

### 4.1 Data Preprocessing

* Categorical features are converted using one-hot encoding
* Data is normalized using MinMaxScaler

### 4.2 Model Architecture

* Input layer based on number of features
* Encoder layers reduce dimensionality
* Bottleneck layer captures important patterns
* Decoder reconstructs the input

### 4.3 Anomaly Detection

* Reconstruction error is calculated using mean squared error
* Threshold is set using the 95th percentile of errors
* Data points with error above threshold are labeled as anomalies

---

## 5. Outputs

### 5.1 anomaly_results.csv

* Original dataset
* Reconstruction error per record
* Anomaly flag (True/False)

### 5.2 top_anomalies.csv

* Top 10 records with highest reconstruction error

### 5.3 reconstructed_data.csv

* Reconstructed feature values from the autoencoder

---

## 6. Model Evaluation

* Mean reconstruction error is printed during execution
* Threshold-based anomaly detection is used
* Works without labeled data (unsupervised learning)

---

## 7. Running the Project

### 7.1 Install Dependencies

```
pip install -r requirements.txt
```

### 7.2 Run Training

```
python src/train.py
```

---

## 8. CI/CD Pipeline

* Implemented using GitHub Actions
* Automatically runs on push to main branch
* Installs dependencies and executes training script
* Generates output files

---

## 9. Docker Support

Docker is included for environment consistency and future deployment.


---

## 10. Conclusion

The project demonstrates how autoencoders can be used for anomaly detection by learning normal data patterns and identifying deviations through reconstruction error.
