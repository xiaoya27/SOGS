#!/usr/bin/env python3
"""
Test script for SOGS (Second-Order Gaussian Splatting) implementation.
This script verifies the core components work correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the Scaffold-GS directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_second_order_statistics():
    """Test the second-order statistics computation."""
    print("=" * 60)
    print("Testing Second-Order Statistics Computation")
    print("=" * 60)
    
    from scene.gaussian_model import GaussianModel
    
    # Create a GaussianModel with SOGS enabled
    feat_dim = 16  # Reduced from 32 as per SOGS paper
    n_offsets = 10
    num_eigenvectors = 2
    
    print(f"Creating GaussianModel with feat_dim={feat_dim}, num_eigenvectors={num_eigenvectors}")
    
    model = GaussianModel(
        feat_dim=feat_dim,
        n_offsets=n_offsets,
        voxel_size=0.01,
        update_depth=3,
        update_init_factor=16,
        update_hierachy_factor=4,
        use_feat_bank=False,
        appearance_dim=0,
        ratio=1,
        add_opacity_dist=False,
        add_cov_dist=False,
        add_color_dist=False,
        use_second_order=True,
        num_eigenvectors=num_eigenvectors
    )
    
    print(f"✓ GaussianModel created successfully")
    print(f"  - use_second_order: {model.use_second_order}")
    print(f"  - num_eigenvectors: {model.num_eigenvectors}")
    print(f"  - mlp_feature_aug: {len(model.mlp_feature_aug)} MLPs")
    
    # Simulate anchor features
    N = 1000  # Number of anchors
    model._anchor_feat = nn.Parameter(torch.randn(N, feat_dim).cuda())
    
    print(f"\nSimulated {N} anchors with {feat_dim}-dim features")
    
    # Compute second-order statistics
    print("\nComputing second-order statistics...")
    model.compute_second_order_statistics()
    
    assert model.top_eigenvectors is not None, "Eigenvectors should be computed"
    assert model.top_eigenvectors.shape == (feat_dim, num_eigenvectors), \
        f"Expected shape ({feat_dim}, {num_eigenvectors}), got {model.top_eigenvectors.shape}"
    assert model._second_order_computed, "Second order should be marked as computed"
    
    print(f"✓ Second-order statistics computed successfully")
    print(f"  - top_eigenvectors shape: {model.top_eigenvectors.shape}")
    
    # Test feature augmentation
    print("\nTesting feature augmentation...")
    test_feat = torch.randn(100, feat_dim).cuda()  # 100 visible anchors
    augmented_feat = model.get_augmented_features(test_feat)
    
    expected_dim = feat_dim * (1 + num_eigenvectors)
    assert augmented_feat.shape == (100, expected_dim), \
        f"Expected shape (100, {expected_dim}), got {augmented_feat.shape}"
    
    print(f"✓ Feature augmentation works correctly")
    print(f"  - Input shape: {test_feat.shape}")
    print(f"  - Output shape: {augmented_feat.shape}")
    print(f"  - Augmentation factor: {augmented_feat.shape[1] / test_feat.shape[1]:.1f}x")
    
    return True


def test_selective_gradient_loss():
    """Test the selective gradient loss computation."""
    print("\n" + "=" * 60)
    print("Testing Selective Gradient Loss")
    print("=" * 60)
    
    # Import the loss function
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # We need to import after adding to path
    import importlib
    import train
    importlib.reload(train)
    from train import selective_gradient_loss
    
    # Create dummy rendered and GT images
    H, W = 256, 256
    rendered = torch.rand(3, H, W).cuda()
    gt = torch.rand(3, H, W).cuda()
    
    print(f"Testing with image size: {H}x{W}")
    
    # Compute loss
    loss = selective_gradient_loss(rendered, gt)
    
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    print(f"✓ Selective gradient loss computed successfully")
    print(f"  - Loss value: {loss.item():.6f}")
    
    # Test gradient flow
    rendered.requires_grad_(True)
    loss = selective_gradient_loss(rendered, gt)
    loss.backward()
    
    assert rendered.grad is not None, "Gradient should flow through"
    print(f"✓ Gradient flow verified")
    
    return True


def test_mlp_dimensions():
    """Test that MLP dimensions are correct with augmented features."""
    print("\n" + "=" * 60)
    print("Testing MLP Input Dimensions")
    print("=" * 60)
    
    from scene.gaussian_model import GaussianModel
    
    feat_dim = 16
    num_eigenvectors = 2
    aug_feat_dim = feat_dim * (1 + num_eigenvectors)  # 16 * 3 = 48
    
    model = GaussianModel(
        feat_dim=feat_dim,
        n_offsets=10,
        voxel_size=0.01,
        update_depth=3,
        update_init_factor=16,
        update_hierachy_factor=4,
        use_feat_bank=False,
        appearance_dim=0,
        ratio=1,
        add_opacity_dist=False,
        add_cov_dist=False,
        add_color_dist=False,
        use_second_order=True,
        num_eigenvectors=num_eigenvectors
    )
    
    # Check MLP input dimensions
    # mlp_opacity input: aug_feat_dim + 3 (view direction)
    opacity_in_features = model.mlp_opacity[0].in_features
    expected_opacity_in = aug_feat_dim + 3
    print(f"mlp_opacity input: {opacity_in_features}, expected: {expected_opacity_in}")
    assert opacity_in_features == expected_opacity_in, \
        f"mlp_opacity input mismatch: {opacity_in_features} vs {expected_opacity_in}"
    
    # mlp_cov input: aug_feat_dim + 3
    cov_in_features = model.mlp_cov[0].in_features
    expected_cov_in = aug_feat_dim + 3
    print(f"mlp_cov input: {cov_in_features}, expected: {expected_cov_in}")
    assert cov_in_features == expected_cov_in, \
        f"mlp_cov input mismatch: {cov_in_features} vs {expected_cov_in}"
    
    # mlp_color input: aug_feat_dim + 3 + appearance_dim (0)
    color_in_features = model.mlp_color[0].in_features
    expected_color_in = aug_feat_dim + 3 + 0
    print(f"mlp_color input: {color_in_features}, expected: {expected_color_in}")
    assert color_in_features == expected_color_in, \
        f"mlp_color input mismatch: {color_in_features} vs {expected_color_in}"
    
    print(f"\n✓ All MLP dimensions are correct for SOGS")
    print(f"  - Original feat_dim: {feat_dim}")
    print(f"  - Augmented feat_dim: {aug_feat_dim}")
    
    return True


def test_memory_comparison():
    """Compare memory usage between original and SOGS."""
    print("\n" + "=" * 60)
    print("Memory Comparison: Scaffold-GS vs SOGS")
    print("=" * 60)
    
    N = 100000  # Number of anchors (typical scene)
    
    # Original Scaffold-GS (feat_dim=32)
    original_feat_dim = 32
    original_feat_memory = N * original_feat_dim * 4  # 4 bytes per float32
    
    # SOGS (feat_dim=16, M=2 eigenvectors)
    sogs_feat_dim = 16
    num_eigenvectors = 2
    sogs_feat_memory = N * sogs_feat_dim * 4
    
    # Additional SOGS overhead (constant, not per-anchor)
    eigenvector_memory = sogs_feat_dim * num_eigenvectors * 4  # D x M
    covariance_memory = sogs_feat_dim * sogs_feat_dim * 4  # D x D (temporary)
    mlp_params = 2 * (2 * sogs_feat_dim * sogs_feat_dim + sogs_feat_dim)  # 2 MLPs
    sogs_overhead = eigenvector_memory + mlp_params * 4
    
    print(f"Anchors: {N:,}")
    print(f"\nOriginal Scaffold-GS (feat_dim={original_feat_dim}):")
    print(f"  - Anchor features: {original_feat_memory / 1024 / 1024:.2f} MB")
    
    print(f"\nSOGS (feat_dim={sogs_feat_dim}, M={num_eigenvectors}):")
    print(f"  - Anchor features: {sogs_feat_memory / 1024 / 1024:.2f} MB")
    print(f"  - Overhead (eigenvectors + MLPs): {sogs_overhead / 1024:.2f} KB")
    print(f"  - Total: {(sogs_feat_memory + sogs_overhead) / 1024 / 1024:.2f} MB")
    
    savings = (original_feat_memory - sogs_feat_memory) / original_feat_memory * 100
    print(f"\n✓ Memory savings: {savings:.1f}%")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("SOGS Implementation Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, some tests may fail")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    tests = [
        ("Second-Order Statistics", test_second_order_statistics),
        ("Selective Gradient Loss", test_selective_gradient_loss),
        ("MLP Dimensions", test_mlp_dimensions),
        ("Memory Comparison", test_memory_comparison),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {name}")
        if error:
            print(f"    Error: {error}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! SOGS implementation is ready.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
