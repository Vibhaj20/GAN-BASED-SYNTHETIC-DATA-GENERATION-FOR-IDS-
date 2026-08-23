# GAN Based Synthetic Data Generation for Imbalanced Dataset in Intrusion Detection System

Final year project (VTU, BGSCET, 2023–2027) addressing class imbalance in network intrusion detection datasets using a dual balancing strategy: KNN-based undersampling for majority classes and GAN-based oversampling for minority attack classes.

**Guide:** Mr. Chethan, Assistant Professor, Dept. of CSE

## Team

| Member |  Role |
|---|---|
| Bindushree D | Data preprocessing and balancing |
| Keerthi BR | Classifier training and evaluation |
| Sahana TH |  Documentation and demo interface |
| Vibha J |  GAN implementation lead |

## Problem Statement

Network intrusion datasets like NSL-KDD are severely imbalanced — common traffic (Normal, DoS) vastly outnumbers rare but dangerous attack types (U2R, R2L). Classifiers trained naively on this data achieve high overall accuracy while being nearly blind to the rare classes that matter most for security. This project addresses that imbalance through a dual balancing strategy, then evaluates downstream classifier performance across four models.

## Methodology

1. **Preprocessing** — cleaning, encoding, scaling (Bindu)
2. **Majority-class undersampling** — KNN-based (NearMiss) reduction of the Normal class (Bindu)
3. **Minority-class oversampling** — GAN-based synthetic sample generation for DoS, Probe, R2L, U2R (Vibha)
   - Vanilla GAN baseline (stability check)
   - WCGAN (Wasserstein GAN with Gradient Penalty)
   - ACGAN (Auxiliary Classifier GAN)
   - Comparative evaluation → per-class architecture selection
4. **Classifier evaluation** — Decision Tree, XGBoost, CNN, AutoGluon (Keerthi)
5. **Documentation & demo interface** (Sahana)

## Datasets

- **NSL-KDD** — primary dataset, pipeline complete through final dataset assembly
- **UNSW-NB15** — second benchmark dataset for generalizability comparison, not yet started

> Note: raw and processed data files (CSVs, model weights) are not tracked in this repository due to size. They are shared with the team via Google Drive. See `data/` folder structure below for what's expected locally.

## Key Results So Far (NSL-KDD)

Comparative mean absolute difference (real vs. synthetic feature means) across three GAN architectures:

| Class | Vanilla GAN | WCGAN | ACGAN | Best |
|---|---|---|---|---|
| U2R | 0.0281 | 0.0206 | **0.0158** | ACGAN |
| R2L | 0.0068 | **0.0051** | 0.0337 | WCGAN |
| Probe | 0.0196 | **0.0088** | 0.0225 | WCGAN |
| DoS | 0.0200 | **0.0068** | 0.0304 | WCGAN |

**Finding:** ACGAN's shared conditional architecture benefits the most data-scarce class (U2R) via implicit cross-class knowledge transfer, while WCGAN's dedicated per-class models perform better on classes with sufficient data. Final augmentation uses architecture selected per class accordingly.

**Final augmented training set:** 174,604 records — Normal/DoS/Probe/R2L balanced to 42,401 each; U2R conservatively boosted to 5,000 (avoiding an extreme ~213x oversampling ratio from only 198 real samples).

Full methodology, hyperparameters, and dated progress log: see [`GAN_Work_Log.md`](./GAN_Work_Log.md).

## Repository Structure

```
├── vanilla_gan_baseline.py       # Vanilla GAN baseline (all 4 classes)
├── wcgan_baseline.py             # WCGAN / WGAN-GP implementation
├── acgan_baseline.py             # ACGAN implementation (joint training)
├── mode_collapse_check.py        # Diagnostic: synthetic sample diversity check
├── knn_undersample_normal.py     # KNN-based undersampling of Normal class
├── merge_final_dataset.py        # Final dataset assembly script
├── GAN_Work_Log.md               # Dated implementation log
├── .gitignore
└── data/                         # NOT tracked in git — see Google Drive
    ├── gan_input_*.csv
    ├── X_train.csv / X_test.csv
    ├── y_train_*.csv / y_test_*.csv
    └── scaler.pkl / label_encoder.pkl
```

## Setup

```bash
pip install torch pandas numpy matplotlib scikit-learn imbalanced-learn
```

Place Bindu's preprocessed data files in a local `data/` folder (not tracked by git — obtain from shared Drive) before running any script.

## Running the Pipeline

```bash
# 1. Confirm GAN training stability
python vanilla_gan_baseline.py

# 2. Train WCGAN (best for R2L, Probe, DoS)
python wcgan_baseline.py

# 3. Check for mode collapse on smallest class
python mode_collapse_check.py

# 4. Train ACGAN (best for U2R)
python acgan_baseline.py

# 5. Undersample majority class
python knn_undersample_normal.py

# 6. Assemble final balanced dataset
python merge_final_dataset.py
```

## Status

- [x] Preprocessing complete (NSL-KDD)
- [x] KNN undersampling complete (NSL-KDD)
- [x] Vanilla GAN baseline — stable across all classes
- [x] WCGAN implementation and evaluation
- [x] ACGAN implementation and evaluation
- [x] Comparative analysis and per-class architecture selection
- [x] Mode collapse diagnostic (U2R)
- [x] Final dataset assembled and handed off to classifier pipeline
- [ ] Classifier training and evaluation (in progress — Keerthi)
- [ ] Full pipeline replication on UNSW-NB15
- [ ] Final comparative report across both datasets
