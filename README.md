# Hyperspectral Disease Detection [![starline](https://starlines.qoo.monster/assets/USER)](https://github.com/qoomon/starline)

You can use this link to open dataset in MATLAB: [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=lishaangral/hyperspectral-disease-detection)


* This repository hosts the complete pipeline for **Hyperspectral Disease Detection** in plants using a deep learning framework that combines **3D Convolutional Autoencoders** and **LSTM networks**.
The project aims to detect and classify plant health conditions by leveraging high-dimensional hyperspectral reflectance data.

* The workflow includes data acquisition, preprocessing, model training, and evaluation, built to support both research and scalable deployment.

---

## Dataset Information

The dataset used in this project is derived from the **Hyperspectral Plant Imaging Dataset** (Dataverse DOI: 10.57745/R6AMN3).

A **prototype subset** focusing on *Plant 2* was extracted, preprocessed, and structured for model development and analysis.

For detailed dataset structure, and file specifications,
**refer to the README located in the `/DATASET` folder.**

---

## Project Structure

```
hyperspectral-disease-detection/
├── DATASET/
│   └── PROTOTYPE/                  → Preprocessed Plant 2.1 data (see dataset README)
├── notebooks/                      → Jupyter notebooks for preprocessing & model training
├── models/                         → 3D CNN Autoencoder + LSTM model definitions
├── scripts/                        → Utility scripts (data loading, training loops, etc.)
├── prototype_data_download.py      → Dataverse download and preprocessing script
├── requirements.txt                → Python dependencies
└── README.md                       → Project overview (this file)
```

## Objectives

1. Implement a deep learning-based approach to **detect early-stage plant disease** from hyperspectral imagery.
2. Develop an **end-to-end preprocessing pipeline** to handle raw `.hdr`, `.dat`, `.png`, and `.csv` hyperspectral data.
3. Build a **3D Convolutional Autoencoder** to extract meaningful spectral–spatial features.
4. Integrate an **LSTM module** for temporal sequence learning across multiple observation days.
5. Validate and visualize the model’s ability to identify disease progression in plants.

---

## Current Progress
```
✔ Prototype dataset extracted and standardized (Plant 2 subset)
✔ Preprocessing pipeline implemented and verified
✔ Initial model architecture design finalized (3D CNN + LSTM hybrid)
⧗ Training pipeline integration and performance evaluation (in progress)
⧗ Visualization and interpretability modules (planned)
```

## Technical Highlights

* **Language:** Python
* **Libraries:** NumPy, TensorFlow/PyTorch, spectral, tqdm, matplotlib, scikit-learn
* **Architecture:** 3D CNN Autoencoder + LSTM sequence model
* **Environment:** Jupyter Notebook + local/remote server execution
* **Dataset Source:** Dataverse API (10.57745/R6AMN3)

---

<!-- ## Usage

1. Clone the repository:
   git clone [https://github.com/lishaangral/hyperspectral-disease-detection.git](https://github.com/lishaangral/hyperspectral-disease-detection.git)

2. Install required packages:
   pip install -r requirements.txt

3. (Optional) Reconstruct the dataset locally:
   python prototype_data_download.py

4. Run preprocessing and visualization notebooks inside `/notebooks`.

--- -->

## Citation

If you use this repository or its derived dataset in your work, please cite:

Hyperspectral Plant Imaging Dataset (2022)
Entrepôt de Données de la Recherche - Data.gouv.fr
DOI: 10.57745/R6AMN3

<!-- ---

## Maintainer

Lisha Angral
Project: Hyperspectral Disease Detection — 3D CNN Autoencoder + LSTM Framework
GitHub: [https://github.com/lishaangral/hyperspectral-disease-detection](https://github.com/lishaangral/hyperspectral-disease-detection) -->

