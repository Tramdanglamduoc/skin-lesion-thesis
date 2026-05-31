# GitHub Repository Name: skin-lesion-thesis

# Using Large-Scale Datasets to Enhance Diagnosis in Limited Multi-Spectral Skin Imaging

**Bachelor Thesis — Riga Technical University, 2026**  
**Author:** Ngoc Bao Tram Tran (Student ID: 231ADB294)  
**Supervisor:** Dr.sc.ing., Associate Professor Dmitrijs Bļizņuks  
**Faculty:** Computer Science, Information Technology and Energy — Institute of Applied Computer Systems

---

## Overview

This repository contains the full source code for the bachelor thesis, which investigates how knowledge learned from large-scale public RGB dermoscopic datasets can be transferred to a smaller proprietary multispectral skin lesion dataset.

The thesis addresses a four-class skin lesion classification task covering:

| Label | Class | Description |
|-------|-------|-------------|
| `bcc` | Basal cell carcinoma | Most common skin cancer |
| `bkl` | Benign keratosis | Solar lentigines / seborrheic keratoses |
| `mel` | Melanoma | Malignant melanocytic lesion |
| `nv` / `nevus` | Melanocytic nevus | Benign mole |

Two complementary strategies are combined:

1. **Multimodal learning** — MultiModalSkinNet fuses dermoscopic image features (EfficientNet-B0 backbone) with structured clinical metadata (age, sex, lesion localisation) via a lightweight MLP branch.
2. **Multistage transfer learning** — a single backbone is progressively adapted from ImageNet → HAM10000 (RGB dermoscopy) → proprietary multispectral dataset.

---

## Repository Structure

```
skin-lesion-thesis/
├── HAM10000-source_model-2nd_stage/      # Stage 2: pretraining on HAM10000
│   ├── MultiModalSkinNet_EfficientnetB0.ipynb
│   └── ImageOnlySkinNet_EfficientnetB0.ipynb
│
├── Proprietrary-target_mode-3rd_stage/   # Stage 3: transfer to multispectral dataset
│   ├── Get_the_structure.ipynb
│   ├── Channel_Orientation_Alignment.ipynb
│   ├── Hair_Removal.ipynb
│   ├── Split_data.ipynb
│   ├── Image_Preprocessing.ipynb
│   ├── EfficientNetB0-HAM10000_weights.ipynb
│   ├── EfficientNetB0-ImageNet1K_weights.ipynb
│   ├── Proprietrary-target-dataset-description.md
│   ├── excluded_lesions.csv
│   ├── manifest.csv
│   ├── orientation_report_final_v2.csv
│   └── folds_5cv.pkl
│
└── README.md
```

---

## Datasets

### HAM10000 (Public)

HAM10000 is a public dermoscopic image dataset of 10,015 images across 7,470 unique lesions. This thesis uses a 4-class subset (bcc, bkl, mel, nv), yielding 9,431 images across 7,071 lesions.

- **Harvard Dataverse:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **Kaggle:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Images are standard white-light RGB dermoscopic captures (600×450 px). Metadata includes age, sex, and lesion localisation, used as inputs to the metadata branch in MultiModalSkinNet.

### Proprietary Multispectral Dataset (Not Shared)

The proprietary dataset was provided by the Institute of Applied Computer Systems, Riga Technical University, and the Institute of Atomic Physics and Spectroscopy, University of Latvia. It contains approximately 1,774 lesions captured by a portable multispectral device with narrow-band LEDs at four spectral bands:

| Channel | Wavelength | Description |
|---------|-----------|-------------|
| G | 526 nm | Green reflectance — superficial skin structures |
| R | 663 nm | Red reflectance — complementary spectral band |
| IR | 964 nm | Near-infrared — deeper skin layers |
| UV / UV10–UV29 | 405 nm | Autofluorescence decay sequence (not used in this thesis) |

This work uses only the G, R, and IR channels. After quality filtering (9 excluded for missing channels, 3 excluded by manual inspection for corrupted captures), **1,762 lesions** were retained for training.

The proprietary dataset is clinical data and is not publicly shared. The code is provided in full so the pipeline can be inspected and reused on any equivalent multispectral acquisition.

---

## Model Architectures

### MultiModalSkinNet (HAM10000 pretraining)

The primary architecture, combining image and metadata modalities.

**Image branch:** A single shared EfficientNet-B0 backbone with its first convolutional layer adapted from 3-channel to 1-channel input (pretrained RGB weights averaged across channels). Each of the three decomposed grayscale channels (R, G, B from HAM10000; R, G, IR from the multispectral dataset) is processed independently by this shared backbone, producing a 1,280-dim feature vector per channel. The three vectors are concatenated to 3,840 dims, followed by Dropout(0.5).

**Metadata branch:** A two-layer MLP trained from scratch. Input: 17 features (age + one-hot encoded sex and localisation). Architecture: `Linear(17→64) → LayerNorm → ReLU → Dropout(0.4) → Linear(64→32) → ReLU`. Output: 32-dim embedding.

**Fusion and classification:** Late fusion by concatenation of image and metadata features (3,872 dims total). Classifier: `Dropout(0.5) → Linear(3872→256) → ReLU → Dropout(0.5) → Linear(256→4)`.

### ImageOnlySkinNet (HAM10000 ablation baseline)

Identical image branch to MultiModalSkinNet (same shared EfficientNet-B0 backbone). No metadata branch. Classifier: `Dropout(0.5) → Linear(3840→256) → ReLU → Dropout(0.5) → Linear(256→4)`. 

ImageOnlySkinNet also serves as the **target model architecture** for Stage 3, since the proprietary multispectral dataset's metadata was excluded from training due to insufficient coverage and missing diagnostic fields.

---

## Stage 2 — HAM10000 Pretraining (`HAM10000-source_model-2nd_stage/`)

### `MultiModalSkinNet_EfficientnetB0.ipynb`

Trains MultiModalSkinNet on HAM10000 using 5-fold stratified group cross-validation (`StratifiedGroupKFold`, `random_state=42`, grouped by `lesion_id` to prevent data leakage).

**Data preprocessing:**
- Hair removal: blackhat morphological transform (7×7 kernel, threshold 20), Navier-Stokes inpainting (`cv2.INPAINT_NS`, radius 5) applied only when hair pixel count ≥ 500.
- RGB channel decomposition: each image split into R, G, B grayscale channels to simulate multispectral narrow-band input.
- Metadata: mean imputation for missing age values; one-hot encoding for sex and localisation; StandardScaler normalization for age.
- Single-channel normalization: `mean=0.449, std=0.226`.
- Class imbalance: `WeightedRandomSampler` (1× epoch size) + two-tier augmentation (base for all classes, heavier pipeline with random perspective, stronger jitter, random erasing for minority classes bcc, bkl, mel at 70% probability).

**Training — two-phase progressive unfreezing:**
- Phase 1 (5 epochs): entire backbone frozen; metadata branch + classifier trained with Adam, lr `1e-4`.
- Phase 2 (up to 50 epochs): last 2 backbone blocks unfrozen; AdamW with differential LRs — backbone `3e-5`, head `1e-4`, weight decay `2e-4`. `CosineAnnealingWarmRestarts` (T₀=10, T_mult=2, η_min=1e-7). Early stopping: patience=9, min_delta=0.001, monitored on validation macro F1. Gradient clipping: max_norm=1.0.

**Results (5-fold CV on HAM10000):**

| Model | Mean Accuracy | Mean Macro F1 |
|-------|--------------|--------------|
| MultiModalSkinNet | 85.17% ± 0.98% | 74.91% ± 1.20% |
| ImageOnlySkinNet  | 80.49% ± 1.56% | 68.48% ± 1.83% |

MultiModalSkinNet outperforms the image-only baseline by +6.43% macro F1 and +4.68% accuracy, with 34–37% lower cross-fold variance.

### `ImageOnlySkinNet_EfficientnetB0.ipynb`

Trains ImageOnlySkinNet on HAM10000. Backbone weights are loaded from the trained MultiModalSkinNet checkpoint (backbone + pooling + image dropout transferred; metadata branch and classifier discarded). Identical data pipeline and two-phase training to MultiModalSkinNet except only the classifier head is trained in Phase 1.

---

## Stage 3 — Multispectral Transfer (`Proprietrary-target_mode-3rd_stage/`)

### Preprocessing Pipeline (run in order)

**`Explore_the_dataset_-_New.ipynb`** *(not included in repo)*  
Initial dataset exploration. Scans all lesion folders, checks for complete R/G/IR channel triplets, and reports missing or incomplete lesions. Identified 9 lesions excluded from subsequent steps.


**`Get_the_structure.ipynb`**  
Input: `skin/` (raw dataset) → Output: `skin_clean_R_G_IR/`

Restructures the raw dataset into a clean, standardized layout. Copies and renames channel images to `R.png`, `G.png`, `IR.png`. Handles the two-level folder structure of `bkl` (which has an extra intermediate grouping level L81, L81.4, L81.4_L81, L82) by flattening it into the lesion name as `<group>__<lesion_id>`. Skips and logs any lesion missing a channel. Produces `manifest.csv` and `excluded_lesions.csv`.

Result: 1,765 valid lesions (mel 538, nevus 527, bcc 450, bkl 250); 9 excluded.


**`Channel_Orientation_Alignment.ipynb`**  
Input: `skin_clean_R_G_IR/` → Output: `skin_clean_R_G_IR_aligned_1/`

Corrects rotational misalignment between spectral channels. Automated alignment uses a multi-feature similarity score (directional Sobel gradients, edge orientation histograms, quadrant intensity statistics, corner marker density) to select the best of four 90° rotations for each G and IR channel relative to R as reference. Saves `orientation_report_final_v2.csv`. After automated alignment, all 1,765 images were manually reviewed and 3 additional lesions with severe image quality issues (complete channel dropout or extreme noise) were excluded.

Result: 1,762 lesions (mel 537, nevus 526, bcc 449, bkl 250).


**`Hair_Removal.ipynb`**  
Input: `skin_clean_R_G_IR_aligned_1/` → Output: `skin_clean_R_G_IR_aligned_1_hairless/`

Per-channel hair removal using blackhat morphological transform + Telea inpainting (`cv2.INPAINT_TELEA`). Channel-specific parameters:

| Channel | Kernel | Threshold | Min hair pixels | Inpaint radius |
|---------|--------|-----------|-----------------|----------------|
| R | 9×9 | 6 | 2,500 | 5 |
| G | 9×9 | 16 | 3,000 | 5 |
| IR | 11×11 | 25 | 7,000 | 6 |

Connected component analysis removes small noise objects (min area 150 px for G, 200 px for IR). Images below the minimum hair pixel threshold are left unchanged.


**`Split_data.ipynb`**  
Input: `skin_clean_R_G_IR_aligned_1_hairless/` → Output: `folds_5cv.pkl`

5-fold stratified group cross-validation using `StratifiedGroupKFold` (`random_state=42`), grouped by `lesion_id` to prevent data leakage. Each fold: ~1,410 train / ~352 validation lesions. Serializes folds as `(fold_idx, train_df, val_df)` tuples.


**`Image_Preprocessing.ipynb`**  
Input: `folds_5cv.pkl` → PyTorch DataLoaders

Defines the `SkinLesionDataset` class and augmentation pipelines (same strategy as HAM10000: base transform for majority classes, heavier transform for minority class `bkl` at 70% probability). `WeightedRandomSampler` set to 2× epoch size to compensate for the smaller dataset. Single-channel normalization: `mean=0.449, std=0.226`.


### Training Notebooks

**`EfficientNetB0-HAM10000_weights.ipynb`**  
Trains ImageOnlySkinNet on the proprietary multispectral dataset, initialized with backbone weights from the best MultiModalSkinNet checkpoint trained on HAM10000 (best val F1: 76.52% on HAM10000). Backbone, global average pooling, and image dropout are transferred; classifier is randomly reinitialized.

**`EfficientNetB0-ImageNet1K_weights.ipynb`**  
Same architecture and training pipeline as above, but backbone initialized from standard ImageNet-1K pretrained EfficientNet-B0 weights (`EfficientNet_B0_Weights.DEFAULT`). Serves as the baseline comparison.

Both notebooks use identical two-phase training:
- Phase 1 (5 epochs): backbone fully frozen; classifier trained with Adam, lr `1e-3`.
- Phase 2 (up to 50 epochs): entire backbone unfrozen; AdamW with differential LRs — backbone `1e-4`, classifier `3e-4`, weight decay `1e-3`. `CosineAnnealingWarmRestarts` (T₀=10, T_mult=2, η_min=1e-6). Early stopping: patience=12, min_delta=0.001, monitored on validation macro F1 across both phases combined.

**Transfer learning results (5-fold CV on proprietary dataset):**

| Initialization | Mean Accuracy | Mean Macro F1 |
|----------------|--------------|--------------|
| HAM10000 pretrained | 81.50% ± 2.46% | 79.78% ± 2.74% |
| ImageNet-1K only    | 79.11% ± 2.38% | 77.27% ± 2.86% |

HAM10000 pretraining provides consistent gains (+2.51% F1, +2.39% accuracy) across 4 of 5 folds, confirming that domain-specific intermediate pretraining improves feature transferability from RGB dermoscopy to multispectral imaging.

---

## Computing Environments

| Stage | Platform | GPU |
|-------|----------|-----|
| HAM10000 pretraining | Google Colab | NVIDIA A100-SXM4-80GB |
| Multispectral transfer | Local workstation | NVIDIA GeForce RTX 4090 (25.4 GB) |

---

## Citation

If you use this code, please cite the thesis:

> Tran, N. B. T. (2026). *Using Large-Scale Datasets to Enhance Diagnosis in Limited Multi-Spectral Skin Imaging*. Bachelor Thesis, Riga Technical University.

---

## Source Code

The source code for this thesis is available at: https://github.com/Tramdanglamduoc/skin-lesion-thesis

The repository contains everything used in this work:
- The HAM10000 pretraining stage, including both MultiModalSkinNet (image + metadata) and ImageOnlySkinNet (image only).
- The transfer learning stage on the proprietary multispectral dataset, covering both initialization strategies: ImageNet-1K only weights and HAM10000-pretrained weights.
- Data preprocessing, fold splitting, training, and evaluation scripts for both stages.

Regarding the datasets: the HAM10000 dataset is public and can be downloaded from the Harvard Dataverse or Kaggle. The proprietary multispectral dataset is not shared, since it is clinical data and access is restricted. The code is still provided in full, so the pipeline can be inspected and reused on any equivalent multispectral acquisition.
