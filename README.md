# 🧠 NeuroAge – Brain Age Prediction & Alzheimer's Disease Assessment

## 📌 Problem

Early detection of neurodegenerative changes is challenging and often requires expert interpretation of brain MRI scans.

NeuroAge aims to estimate biological brain age from structural MRI and quantify accelerated brain aging using Brain Age Gap (BAG), providing a research-oriented tool for brain health assessment.

---

## 🔧 Methodology

* Collected and analyzed MRI scans from the ADNI dataset

* Preprocessed T1-weighted MRI scans using MONAI

* Developed 3D Deep Learning models using PyTorch

* Performed Alzheimer's Disease classification (CN, MCI, AD)

* Applied Transfer Learning for Brain Age Regression

* Predicted Brain Age from MRI scans

* Computed Brain Age Gap (BAG):

  ```
  BAG = Predicted Brain Age − Chronological Age
  ```

* Integrated trained models into a mobile application

* Deployed inference pipeline using Hugging Face

---

## 📊 Results

* Successfully trained deep learning models on ADNI MRI data
* Automated Brain Age prediction from structural MRI scans
* Generated Brain Age Gap measurements for CN, MCI, and AD subjects
* Developed NeuroAge Android application for MRI-based analysis
* Research manuscript currently under preparation

---

## 📱 Mobile Application Features

* Upload MRI scans (.nii/.nii.gz)
* Brain Age estimation
* Brain Age Gap calculation
* Analysis history tracking
* Research-oriented brain health assessment

---

## 🛠️ Tech Stack

* Python
* PyTorch
* MONAI
* NumPy
* Pandas
* Scikit-learn
* Hugging Face
* Android Studio
* Java

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python inference.py --image sample_scan.nii.gz
```

---

## ⚠️ Disclaimer

This project is intended for research and educational purposes only and is not a medical diagnostic tool.

---

## 👨‍💻 Author

**Gowri Sankhar Saravanan**

MSc Health Data Science

Interests:

* Medical AI
* Neuroimaging
* Brain Age Prediction
* Alzheimer's Disease Research
* Healthcare Analytics
