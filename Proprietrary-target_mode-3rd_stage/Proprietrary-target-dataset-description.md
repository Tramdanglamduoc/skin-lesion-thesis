# Proprietary Target Dataset — 3rd Stage Pipeline

This folder contains the full preprocessing and training pipeline for the **proprietary multispectral skin lesion dataset** (referred to as the `skin` dataset), used as the target domain in the transfer learning stage of the thesis.

---

## Dataset Overview

The raw dataset (`skin/`) contains four diagnostic classes stored as folders:

```
skin/
├── bcc/
├── mel/
├── nevus/
└── bkl/
```

### Folder structure by class

**`bcc`, `mel`, `nevus`** follow a two-level hierarchy:

```
skin/<class>/<lesion_id>/
    ├── in/
    │   ├── R.png       # Red channel image
    │   ├── G.png      # Green channel image
    │   └── IR.png      # Infrared channel image
    ├── metadata/           # (present in some lesions)
    │   ├── metadata.csv
    │   └── metadata2.csv
    └── search-res-info.csv # (present in some lesions)
```

Example lesion folder name: `2_1000_922_in_8252___2023-03-22_10-36-00`

**`bkl`** has one additional intermediate grouping level (L81, L81.4, L81.4\_L81, L82):

```
skin/bkl/<group>/<lesion_id>/
    └── in/
        ├── R.png
        ├── G.png
        └── IR.png
```

Each lesion contains three multispectral channel images captured at different exposure times. The `metadata/` folder and `search-res-info.csv` are present only in a subset of lesions and were **not used** in model training (only the channel images were retained).

---

## Pipeline Overview

The notebooks should be run in the following order:

```
Explore_the_dataset_-_New.ipynb         [exploratory, not in repo]
    ↓
Get_the_structure.ipynb
    ↓
Channel_Orientation_Alignment.ipynb
    ↓
Hair_Removal.ipynb
    ↓
Split_data.ipynb
    ↓
Image_Preprocessing.ipynb
    ↓
EfficientNetB0-HAM10000_weights.ipynb   ┐  two training experiments
EfficientNetB0-ImageNet1K_weights.ipynb ┘
```

---

## Notebook Descriptions

### `Explore_the_dataset_-_New.ipynb` *(not included in repo)*

Initial dataset exploration. Scans all lesion folders across all four classes (handling the two-level structure of `bkl`), checks which lesions have all three channel images (R, G, IR), and reports any lesions with missing channels or missing `/in/` folders. This step identified 9 incomplete lesions that were subsequently excluded.

---

### `Get_the_structure.ipynb`

**Input:** `skin/` (raw dataset)  
**Output:** `skin_clean_R_G_IR/`

Restructures the raw dataset into a clean, standardized layout. For each lesion that has all three channel images (R, G, IR):

- Copies and renames the channel images to `R.png`, `G.png`, `IR.png` (standardizing away original filenames like `R_4ms.png`).
- For `bkl`, flattens the intermediate group level by encoding it into the lesion name (e.g., `L81.4__<lesion_id>`), so the output maintains a flat `<class>/<lesion_name>/` structure for all classes.
- Skips lesions missing any channel and logs them to `excluded_lesions.csv`.

Produces two CSV files:
- `manifest.csv` — index of all valid lesions with their channel paths.
- `excluded_lesions.csv` — list of excluded lesions and the reason.

Final class counts after cleaning: `mel` 538, `nevus` 527, `bcc` 450, `bkl` 250 (total: 1,765 lesions, 9 excluded).

Output structure:

```
skin_clean_R_G_IR/
├── bcc/<lesion_name>/{R.png, G.png, IR.png}
├── mel/<lesion_name>/{R.png, G.png, IR.png}
├── nevus/<lesion_name>/{R.png, G.png, IR.png}
├── bkl/<group>__<lesion_name>/{R.png, G.png, IR.png}
├── manifest.csv
└── excluded_lesions.csv
```

---

### `Channel_Orientation_Alignment.ipynb`

**Input:** `skin_clean_R_G_IR/`  
**Output:** `skin_clean_R_G_IR_aligned_1/`

Corrects channel orientation misalignment across the multispectral images. Because the R, G, and IR channels are captured by different sensors, they may be rotated relative to each other (in multiples of 90°). This notebook aligns the G and IR channels to the R channel (used as reference) for each lesion.

For each lesion, alignment is determined by:

1. Extracting multi-feature descriptors from each channel image — directional Sobel gradients (horizontal and vertical), edge orientation histograms (18 bins), quadrant intensity statistics, corner marker density, and overall gradient maps.
2. Testing all four 90° rotations (identity, 90°, 180°, 270°) of the moving channel against the reference.
3. Selecting the rotation that minimizes a weighted similarity score across all features.
4. Applying the correction only when the improvement over identity exceeds a threshold and confidence is sufficient (both R and IR channels are used jointly to make the correction decision).

Produces `skin_clean_R_G_IR_aligned_1/` with the same folder layout and saves `orientation_report_final_v2.csv` logging the correction decision for each lesion.

Following automated alignment, channel images were visually inspected and three lesions were excluded due to corrupted channel captures (a partially exposed R channel, and two cases of a fully dark G channel caused by sensor failure). This yielded a final aligned dataset of `mel` 537, `nevus` 526, `bcc` 449, `bkl` 250 (total: 1,762 lesions).

---

### `Hair_Removal.ipynb`

**Input:** `skin_clean_R_G_IR_aligned_1/`  
**Output:** `skin_clean_R_G_IR_aligned_1_hairless/`

Removes hair artifacts from channel images using **blackhat morphological transform** followed by inpainting. Each channel is processed independently with tuned parameters:

| Channel | Kernel size | Threshold | Min hair pixels | Inpaint radius |
|---------|-------------|-----------|-----------------|----------------|
| R       | 9×9         | 6         | 2,500           | 5              |
| G       | 9×9         | 16        | 3,000           | 5              |
| IR      | 11×11       | 25        | 7,000           | 6              |

For each channel image, the pipeline:

1. Converts to grayscale and applies the blackhat morphological transform to detect dark elongated structures (hair).
2. Thresholds the blackhat result to produce a binary hair mask.
3. Removes small disconnected components from the mask (noise filtering).
4. Dilates the mask to ensure full coverage over hair strands.
5. Applies `cv2.inpaint` (Telea method) only if the total hair pixel count exceeds the per-channel minimum threshold — lesions with little or no hair are left unchanged.

Output maintains the same directory layout as the input. Saves 1,765 lesions to `skin_clean_R_G_IR_aligned_1_hairless/`.

---

### `Split_data.ipynb`

**Input:** `skin_clean_R_G_IR_aligned_1_hairless/`  
**Output:** `folds_5cv.pkl`

Splits the dataset into 5-fold cross-validation folds using `StratifiedGroupKFold` with `lesion_id` as the group key. This ensures:

- Each lesion appears in exactly one split (train or validation) per fold — no data leakage from the same physical lesion appearing in both.
- Class proportions are approximately preserved across all folds.

Each fold is stored as a tuple `(fold_idx, train_df, val_df)` where each DataFrame contains `lesion_id`, `label`, `class_name`, and `folder_path`. The folds are serialized to `folds_5cv.pkl` for use in subsequent notebooks.

---

### `Image_Preprocessing.ipynb`

**Input:** `folds_5cv.pkl`, `skin_clean_R_G_IR_aligned_1_hairless/`  
**Output:** PyTorch DataLoaders (used directly in training notebooks)

Defines the data augmentation pipeline and the `SkinLesionDataset` class used during training.

**Augmentation strategy:**

- *Base augmentation* (majority classes): random resized crop (scale 0.85–1.0), horizontal/vertical flip, mild color jitter (brightness/contrast ±0.15), light Gaussian noise (std 0.01–0.03, p=0.3).
- *Minority augmentation* (`bkl`, label 1): same as base plus random perspective distortion (p=0.2), stronger color jitter (±0.2), stronger Gaussian noise (std 0.01–0.05, p=0.4), and random erasing (p=0.3).
- *Validation*: resize to 300×300, normalize only.

All transforms use single-channel normalization: `mean=[0.449], std=[0.226]`.

**`SkinLesionDataset`** loads R, G, IR images as grayscale (`PIL` `'L'` mode) from each lesion folder and applies the appropriate transform. During training, minority class samples are augmented with the heavy pipeline at 70% probability; all others use the base pipeline. Returns `(img_R, img_G, img_IR, label)` per sample.

**DataLoaders** use `WeightedRandomSampler` for class-balanced batch construction (2× oversampling per epoch with replacement). Validation loaders use standard sequential sampling.

---

### `EfficientNetB0-HAM10000_weights.ipynb`

**Input:** `folds_5cv.pkl`, pre-trained HAM10000 checkpoints (`Weights/EfficientNetB0/checkpoint_fold_*.pth`), `skin_clean_R_G_IR_aligned_1_hairless/`

Trains the **MultiModalSkinNet** on the proprietary dataset using backbone weights pre-trained on the HAM10000 source task (5-fold cross-validation, loaded from checkpoints). This is the primary transfer learning experiment.

Architecture:

- Shared EfficientNet-B0 backbone adapted for single-channel input (first conv layer converted from 3-channel to 1-channel by averaging pretrained RGB weights).
- Three branches — one per channel (R, G, IR) — sharing the same backbone weights, each producing a 1,280-dim feature vector.
- Channel features concatenated to 3,840-dim, fused with a 32-dim metadata MLP output → 3,872-dim combined feature.
- Classifier: `Dropout → Linear(3872, 256) → ReLU → Dropout → Linear(256, 4)`.

Training uses a two-phase strategy:
- **Phase 1:** backbone frozen, only classifier is trained using Adam with lr `1e-3`.
- **Phase 2:** Entire backbone unfrozen, trained with AdamW using differential learning rates — backbone `1e-4`, classifier `3e-4`, weight decay `1e-3`. Learning rate is annealed with `CosineAnnealingWarmRestarts` (T₀=10, T_mult=2, η_min=1e-6).

Scheduler: `CosineAnnealingWarmRestarts`. Optimizer: AdamW with gradient clipping.

---

### `EfficientNetB0-ImageNet1K_weights.ipynb`

**Input:** `folds_5cv.pkl`, `skin_clean_R_G_IR_aligned_1_hairless/`

Identical pipeline and architecture to the HAM10000 experiment, but initializes the backbone from **ImageNet-1K pretrained weights** (`EfficientNet_B0_Weights.DEFAULT`) instead of HAM10000 checkpoints. Serves as the baseline comparison to quantify the benefit of domain-specific pretraining.

---

## Output Dataset Structure (after full pipeline)

```
skin_clean_R_G_IR_aligned_1_hairless/
├── bcc/
│   └── <lesion_name>/
│       ├── R.png
│       ├── G.png
│       └── IR.png
├── mel/   (same structure)
├── nevus/ (same structure)
└── bkl/
    └── <group>__<lesion_name>/
        ├── R.png
        ├── G.png
        └── IR.png
```
