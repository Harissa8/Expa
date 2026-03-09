"""EigenCAM — principal-component activation heatmap (standalone).

How it works:
    1. Given a CNN activation map of shape (B, C, H, W)
    2. Reshape to (C, H*W)
    3. Center the data (subtract per-channel mean)
    4. Run SVD → first right singular vector = dominant spatial pattern
    5. Reshape back to (H, W) and take absolute value
    6. Normalize to [0, 1]

Adversarial patches create abnormally concentrated heatmaps.
Natural objects produce spread-out heatmaps.

No gradients needed → fast and robust.
"""

import torch


def eigencam(activations: torch.Tensor) -> torch.Tensor:
    """Compute EigenCAM heatmaps from CNN activation maps.

    Args:
        activations: (B, C, H, W) tensor from a conv/residual layer

    Returns:
        heatmaps: (B, H, W) tensor, values in [0, 1]
    """
    B, C, H, W = activations.shape
    reshaped   = activations.reshape(B, C, H * W).float()
    heatmaps   = torch.zeros(B, H, W, device=activations.device)

    for i in range(B):
        matrix = reshaped[i]  # (C, H*W)

        # Center across spatial dimension
        matrix = matrix - matrix.mean(dim=1, keepdim=True)

        try:
            _, _, Vh = torch.linalg.svd(matrix, full_matrices=False)
        except torch.linalg.LinAlgError:
            # Fallback if SVD fails: channel-mean map
            heatmaps[i] = activations[i].mean(dim=0).abs()
            continue

        # First right singular vector = first principal spatial pattern
        pc1     = Vh[0]              # (H*W,)
        heatmap = pc1.reshape(H, W).abs()

        # Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-8:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = torch.zeros_like(heatmap)

        heatmaps[i] = heatmap

    return heatmaps
