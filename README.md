# SOGS: Second-Order Gaussian Splatting

Unofficial implementation of **Second-Order Anchor** for Scaffold-GS, based on the paper:
> "SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting" (arXiv:2503.07476)

## Overview

SOGS improves anchor-based 3D Gaussian Splatting by introducing:
1. **Second-Order Anchors**: Uses covariance-based statistics to capture co-variation patterns across anchor feature dimensions
2. **Selective Gradient Loss**: Focuses optimization on difficult-to-render textures and geometries

This enables **superior rendering quality with reduced model size**

## Key Changes

### 1. New Parameters (`arguments/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no_second_order` | `True` | Enable no second-order anchor feature augmentation |
| `--num_eigenvectors` | `2` | Number of eigenvectors M for co-variation patterns |
| `--lambda_sgl` | `0.01` | Weight for selective gradient loss |

### 2. Second-Order Anchor (`scene/gaussian_model.py`)

- **Feature Augmentation MLPs**: M additional MLPs that process eigenvector-guided features
- **`compute_second_order_statistics()`**: Computes covariance matrix and extracts top-M eigenvectors
- **`get_augmented_features()`**: Augments anchor features from dimension D to D×(1+M)

### 3. Renderer Integration (`gaussian_renderer/__init__.py`)

- Calls `get_augmented_features()` when SOGS is enabled
- Feature dimension changes from `feat_dim` to `feat_dim × (1 + num_eigenvectors)`

### 4. Selective Gradient Loss (`train.py`)

- Uses Sobel operators to compute gradient maps
- Applies dynamic region selection to focus on high-error areas
- Loss function: `L = λ₁·L₁ + λ_SSIM·L_SSIM + λ_vol·L_vol + λ_s·L_sgl`

## Usage

### Training with SOGS (Default)

```bash
# SOGS is enabled by default with feat_dim=32
python train.py -s /path/to/dataset -m /path/to/output

# Recommended: Use reduced feat_dim for memory savings
python train.py -s /path/to/dataset -m /path/to/output --feat_dim 16
```

### Training without SOGS (Original Scaffold-GS)

```bash
# Disable SOGS explicitly
python train.py -s /path/to/dataset -m /path/to/output --no_second_order --feat_dim 32
```

### Custom SOGS Configuration

```bash
# Use 3 eigenvectors with higher gradient loss weight
python train.py -s /path/to/dataset -m /path/to/output \
    --feat_dim 12 \
    --num_eigenvectors 3 \
    --lambda_sgl 0.02
```

## Testing the Implementation

Run the test script to verify SOGS is working correctly:

```bash
conda activate feature_3dgs2
python test_sogs.py
```

Expected output:
```
✓ PASSED: Second-Order Statistics
✓ PASSED: Selective Gradient Loss
✓ PASSED: MLP Dimensions
✓ PASSED: Memory Comparison
```

## Memory Efficiency

With 100,000 anchors:

| Configuration | Feature Memory | Savings |
|---------------|---------------|---------|
| Scaffold-GS (feat_dim=32) | 12.21 MB | - |
| SOGS (feat_dim=16, M=2) | 6.10 MB | **50%** |
| SOGS (feat_dim=12, M=2) | 4.58 MB | **62.5%** |

The overhead from eigenvectors and MLPs is negligible (~4 KB).

## Mathematical Background

### Second-Order Statistics (Equations 4-9)

Given anchor features F^a ∈ R^(N×D):

1. **Covariance Matrix**: Σ = (1/(N-1)) × (F^a - μ)^T × (F^a - μ)
2. **Correlation Matrix**: R = A^(-1) × Σ × A^(-1) (standardized)
3. **Eigendecomposition**: R = Q × Λ × Q^T
4. **Select top-M eigenvectors**: P = [P₁, ..., P_M]

### Feature Augmentation (Equation 10-11)

For each anchor feature f^a:
```
f_i^t = MLP_i([P_i, f^a])  for i ∈ [1, M]
output = concat([f^a, f_1^t, ..., f_M^t])
```

### Selective Gradient Loss (Equations 12-17)

```
G'_x = Sobel_x * I'    (rendered gradient)
G_x  = Sobel_x * I     (ground truth gradient)
w_x  = |G'_x - G_x|    (weight map)
L_s  = w_x · l_x + w_y · l_y
```

## File Changes Summary

```
arguments/__init__.py      # +7 lines  (new parameters)
scene/gaussian_model.py    # +204/-65  (second-order anchor)
gaussian_renderer/__init__.py  # +7/-2 (feature augmentation)
train.py                   # +104/-5  (selective gradient loss)
test_sogs.py               # +280     (test script)
```

## Citation

If you use this implementation, please cite both papers:

```bibtex
@article{zhang2025sogs,
  title={SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting},
  author={Zhang, Jiahui and Zhan, Fangneng and Shao, Ling and Lu, Shijian},
  journal={arXiv preprint arXiv:2503.07476},
  year={2025}
}

@inproceedings{lu2024scaffold,
  title={Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={CVPR},
  year={2024}
}
```
