# ResViT: Hybrid CNN-Transformer for Tile-Wise Permuted Image Classification

**Dan Fang** · M.Sc. Machine Learning & Data Science, Reichman University · 2025  
Mentor: Alon Oring

📄 [Read the Full Report (PDF)](./Project_version_4_public_.pdf) · 💻 [Colab Notebook](https://colab.research.google.com/drive/15rRizCaTSsFkGv2MNrZrJCAJwJMQME-9)

---

## Overview

CNNs excel at standard image classification but degrade significantly when images are spatially disrupted — for example, when an image is divided into tiles and those tiles are shuffled. This project investigates how tile-wise permutations affect classification performance, and proposes **ResViT**, a hybrid CNN-Transformer architecture designed to be robust to spatial disorder.

**Key contributions:**
- ResViT: a hybrid model combining a frozen pretrained ResNet50 feature extractor with a custom Transformer Encoder
- Systematic evaluation across 5 permutation types and 8 tile resolutions (9 to 400 tiles)
- A novel **Weighted Disorder Score** metric to rank permutation difficulty without requiring model evaluation
- Grad-CAM and attention map analysis confirming ResViT captures semantics rather than memorizing permutations

---

## Results

### ResViT vs ResNet50 — Random Permutation

| Model | 36 Tiles | 100 Tiles | 225 Tiles | 400 Tiles |
|---|---|---|---|---|
| ResNet50 (baseline) | 93.37% | 79.95% | 66.08% | 58.33% |
| ResViT | 97.79% | 96.81% | 88.41% | 80.14% |
| ResViT (fine-tuned) | **99.25%** | **98.18%** | **90.04%** | **87.62%** |
| Unpermuted benchmark | — | — | — | 98.10% |

ResNet50 drops **35 percentage points** from 36 to 400 tiles under fully random permutation. ResViT (fine-tuned) drops only **11 points**, maintaining near-benchmark accuracy even at extreme spatial disorder.

---

## Architecture

ResViT combines:
- **ResNet50** (frozen pretrained) — local feature extraction
- **Linear projection** — maps ResNet50 feature maps into an embedding space
- **Learnable [CLS] token + positional embeddings**
- **Transformer Encoder blocks** — global reasoning over permuted patch sequences
- **Classification head**

Unlike standard Vision Transformers (ViT) that tokenize raw patches, ResViT processes entire permuted images, making it compatible with externally applied tile-wise permutations.

---

## Permutation Types

Five permutation types tested across 8 resolutions (9, 16, 36, 64, 100, 169, 225, 400 tiles):

| Type | Description |
|---|---|
| **Random** | Fully random shuffle of all tiles |
| **Row Swap** | Tiles shuffled row-wise, column structure preserved |
| **Row-Col Swap** | Rows shuffled, then columns within each row |
| **Partial Random (50%)** | Random 50% of tiles permuted, rest fixed |
| **Random Half** | One spatial half permuted, other half frozen |

Difficulty ranking (least → most disruptive):
**Original < Random Half < Row Swap < Partial < Row-Col < Random**

---

## Permutation Difficulty Metric — Weighted Disorder Score

A novel composite metric combining:
1. **Tile Distance** — average Euclidean displacement of tiles from original positions
2. **Spatial Entropy** — Shannon entropy of relative distance changes between tile pairs

The metric predicts permutation difficulty and model accuracy without requiring model evaluation. A polynomial regression fit (R² validated) shows strong inverse correlation between disorder score and classification accuracy.

---

## Explainability — Grad-CAM & Attention Maps

- **ResNet50** focuses on local regions; attention becomes unfocused under high disorder
- **ResViT** maintains coherent attention on semantic features (fur, ears, limbs) even under fully random permutation — confirming global reasoning rather than permutation memorization

---

## Training Setup

| Parameter | Value |
|---|---|
| Dataset | Kaggle Dogs vs. Cats (25,000 images) |
| Split | 80% train / 20% validation, seed 42 |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 1×10⁻⁴ (fine-tune: 1×10⁻⁵ to 1×10⁻⁷) |
| Batch size | 16 |
| Epochs | 25 (fine-tune: up to 100 with early stopping) |
| Precision | 16-bit mixed |
| Framework | PyTorch Lightning |

---

## Repository Contents

```
├── Dan_Fang_Final_Project_Permuted_Puzzle__version_5.pdf   # Full report
└── README.md
```

Full code and notebooks available in the [Colab Notebook](https://colab.research.google.com/drive/15rRizCaTSsFkGv2MNrZrJCAJwJMQME-9).
