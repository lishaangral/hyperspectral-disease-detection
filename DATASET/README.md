
# Hyperspectral Disease Detection | Prototype Dataset

This folder contains the preprocessed prototype dataset used for model training and validation in the Hyperspectral Disease Detection project.

---

## Folder Overview

``` markdown
DATASET/
└── PROTOTYPE/
├── day_2/
│   └── plant_2.1/
│       ├── 1972.png
│       ├── REFLECTANCE_1972.hdr
│       ├── REFLECTANCE_1972.dat
│       └── pos_Plant2.1.csv
├── day_3/
│   └── plant_2.1/
│       ├── 2011.png
│       ├── REFLECTANCE_2011.hdr
│       ├── REFLECTANCE_2011.dat
│       └── pos_Plant2.1.csv
├──
``` 
Each day_X folder represents hyperspectral imaging data collected on a given day.
Within each day_X folder, only the Plant 2 data has been retained and standardized for this prototype version.

---

## Dataset Origin

The dataset originates from the Dataverse repository hosted at:

[https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/R6AMN3](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/R6AMN3)

Original file DOI used for extraction:
doi:10.57745/XAZQ7O

This dataset includes hyperspectral imagery captured for multiple plants across several growth days.

---

## Preprocessing Workflow Summary

Data download was performed using the Python script:
prototype_data_download.py

Steps:

1. Download the full Dataverse ZIP (~30 GB) using the file DOI.
2. Extract relevant Plant 2 data only, with automatic discrepancy handling:

   * day_2 → plant_2
   * day_3 → plant_21
   * day_15–17 → plant_2.5
3. Normalize folder structure to:
   day_X/plant_2.1/
4. Keep only relevant file types:
   .dat  – Hyperspectral data cube (raw reflectance)
   .hdr  – ENVI header metadata file
   .png  – RGB preview image
   .csv  – Positional or annotation data
5. Rename all .csv files to pos_Plant2.1.csv
6. Delete the 30 GB ZIP archive after extraction to conserve disk space (final prototype dataset ≈ 5–5.5 GB)

---

## Data Specifications

| File Type | Description                           | Typical Size |
| --------- | ------------------------------------- | ------------ |
| .dat      | Binary hyperspectral reflectance cube | 200–250 MB   |
| .hdr      | ENVI header file (metadata for .dat)  | < 1 MB       |
| .png      | RGB preview image                     | 1–5 MB       |
| .csv      | Positional or annotation data         | < 1 MB       |

Total data retained: ~5.5 GB
Original archive size: ~30 GB

---

## Usage Notes

* This prototype dataset focuses exclusively on Plant 2, providing a consistent and representative subset for model training and benchmarking.
* The folder structure and file naming are standardized for seamless integration with preprocessing and deep learning scripts.

---

## GitHub Storage & Large File Handling

To maintain repository performance and comply with GitHub’s 100 MB per-file limit:

All .dat files have been excluded from the GitHub repository as these files are large (≈ 200–250 MB each) and are not necessary for demonstration or structure replication.

If you need the complete dataset, you can:

* Download directly from the Dataverse link above.

---

## Citation

If you use this dataset or its processed version in your work, please cite:

Hyperspectral Plant Imaging Dataset (2022)
Entrepôt de Données de la Recherche – Data.gouv.fr
DOI: 10.57745/R6AMN3
