# 🧠 NeuroAge — AI-Powered Brain Age & Alzheimer's Biomarker Estimation

NeuroAge is an end-to-end deep learning framework that estimates an individual's **biological brain age** from structural MRI scans and uses the resulting **Brain Age Gap (BAG)** as a biomarker for neurodegenerative changes associated with Alzheimer's Disease (AD).

The project spans the full pipeline — from MRI preprocessing and 3D CNN model training to a deployed inference API and a companion Android application.

> ⚠️ **Disclaimer:** This project is intended for **research and educational purposes only** and is **not designed to provide clinical diagnoses**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Mobile Application](#mobile-application)
- [Deployment](#deployment)
- [Project Status](#project-status)
- [Research Significance](#research-significance)
- [License](#license)

---

## Overview

Chronological age doesn't always reflect the true biological condition of the brain. Neurodegenerative disorders like Alzheimer's Disease often accelerate structural brain aging before clinical symptoms become apparent. NeuroAge leverages this idea by training a 3D CNN to predict brain age from T1-weighted MRI scans, then comparing that prediction against chronological age to compute the **Brain Age Gap (BAG)**:

```
BAG = Predicted Brain Age − Chronological Age
```

A positive BAG suggests accelerated brain aging and potential neurodegenerative changes, making it a promising biomarker for early detection.

## Research Motivation

The core hypothesis driving NeuroAge is that BAG should increase progressively across diagnostic groups:

```
CN < MCI < AD
```

Cognitively Normal (CN) subjects are expected to show the smallest brain age gap, while Alzheimer's Disease (AD) subjects are expected to show the largest, with Mild Cognitive Impairment (MCI) subjects falling in between. The project's primary goal is to build a regression model capable of estimating brain age and to statistically validate this BAG pattern across populations.

## Dataset

MRI data is sourced from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**.

| Group | Scans | Mean Age |
|-------|-------|----------|
| Cognitively Normal (CN) | 134 | ≈ 76.8 years |
| Mild Cognitive Impairment (MCI) | 150 | ≈ 76.1 years |
| Alzheimer's Disease (AD) | 122 | ≈ 76.8 years |
| **Total** | **406** | — |

**Modality:** T1-weighted Structural MRI
**Format:** NIfTI (`.nii`, `.nii.gz`)

**Metadata fields:**
- Subject ID
- Diagnosis Group
- Age
- Sex
- MRI Format Information

## Preprocessing Pipeline

MRI preprocessing is built on **MONAI** and **PyTorch**-based medical imaging workflows:

1. MRI loading
2. Orientation standardization
3. Resampling to common voxel spacing
4. Intensity normalization
5. Spatial alignment
6. Tensor conversion
7. Fixed-size volume generation

**Output volume size:** `96 × 96 × 96` voxels
**Lightweight variant:** `80 × 80 × 80` voxels (used in faster experiments)

All scans are standardized into a format suitable for 3D CNN training.

## Project Architecture

NeuroAge follows a **multi-stage learning pipeline**:

### Stage 1 — Alzheimer's Disease Classification
A 3D CNN is trained to distinguish CN vs. AD subjects. This acts as **feature pretraining**, allowing the network to learn disease-related structural patterns that transfer to the brain age task.

### Stage 2 — Brain Age Regression
The pretrained feature extractor is reused, with the classification head replaced by a regression head predicting brain age in years. The model is trained **only on CN subjects**, since healthy aging patterns should be learned before applying the model to diseased populations.

**Evaluation metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation
- R² Score

### Stage 3 — Brain Age Gap Computation
The trained regression model is applied across CN, MCI, and AD subjects to generate predicted brain age and compute BAG for each subject, testing the expected pattern of `CN < MCI < AD`.

### Stage 4 — Statistical Analysis
BAG values are analyzed using group-wise comparisons, box plots, violin plots, scatter plots, correlation analysis, and significance testing to determine whether BAG differs meaningfully across diagnostic groups.

## Tech Stack

**Deep Learning / Data Science:**
- Python
- PyTorch
- MONAI
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

**Training Environment:**
- Google Colab (NVIDIA T4 GPU)

## Mobile Application

**NeuroAge** Android app serves as the user-facing interface for model inference.

**Features:**
- Upload MRI scans
- Send MRI data to the backend inference pipeline
- Predict brain age
- Calculate Brain Age Gap
- Store analysis history locally
- Display assessment results

**Frontend:** Android Studio, Java, XML
**Backend:** Python, Hugging Face Inference API

## Deployment

```
MRI Upload → Backend Processing → Model Inference → Brain Age Prediction → BAG Calculation → Results Returned to Mobile App
```

The trained model is deployed via **Hugging Face**, with the Android app communicating to it through the Hugging Face Inference API.

## Project Status

**✅ Completed**
- Dataset preparation
- MRI preprocessing pipeline
- AD classification model
- Brain age regression pipeline
- Android application development
- Hugging Face deployment

**🚧 In Progress**
- Brain Age Gap statistical analysis
- Research manuscript preparation
- Performance optimization
- Additional validation studies

## Research Significance

NeuroAge aims to provide a non-invasive, AI-based framework for studying brain aging and neurodegeneration, with potential applications in:

- Brain aging research
- Alzheimer's disease research
- Neuroimaging analytics
- Healthcare AI
- Biomarker development
- Clinical decision-support research

## License

*(Add your preferred license here — e.g., MIT, Apache 2.0 — or specify usage restrictions given the research/educational nature of this project.)*

---

*This project is intended for research and educational purposes only and is not designed to provide clinical diagnoses.*
