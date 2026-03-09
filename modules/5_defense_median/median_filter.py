"""Median Filter Defense — standalone implementation.

How it works:
    1. Apply a median filter to the image
       → destroys adversarial patch pixel patterns (high-frequency artifacts)
       → natural textures survive (low-frequency content is preserved)
    2. Run model on BOTH original and filtered image
    3. Compute KL divergence between the two output distributions
    4. Decision rule:
       - class changed    → natural texture sensitivity → CLEAN (not an attack)
       - class same + KL > threshold → patch shifted distribution → ATTACK

Why median filter detects patches:
    Adversarial patches consist of optimized pixel patterns that fool the model.
    These patterns are NOT robust to median filtering — the filter replaces
    each pixel with the median of its neighborhood, breaking up the precise
    pixel values the patch needs.  After filtering, the prediction shifts back
    toward the true class (large KL divergence = suspicious).
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageFilter


_to_tensor = T.ToTensor()

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2470, 0.2435, 0.2616]


def apply_median_filter(image_tensor: torch.Tensor,
                        kernel_size: int = 3) -> torch.Tensor:
    """Apply median filter to a (3, H, W) float tensor in [0, 1].

    PIL MedianFilter replaces each pixel with the median of its
    kernel_size x kernel_size neighbourhood.

    Recommended kernel sizes:
        32×32 images (CIFAR-10): kernel_size=3  (3×3)
        224×224 images (camera): kernel_size=5  (5×5)
    """
    arr = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil = pil.filter(ImageFilter.MedianFilter(size=kernel_size))
    return _to_tensor(pil)


def normalize_batch(tensors: torch.Tensor) -> torch.Tensor:
    """Normalize a (B, 3, H, W) tensor with CIFAR-10 mean/std."""
    m = torch.tensor(MEAN, dtype=tensors.dtype).view(1, 3, 1, 1)
    s = torch.tensor(STD,  dtype=tensors.dtype).view(1, 3, 1, 1)
    return (tensors - m) / s


def kl_divergence(orig_probs: torch.Tensor,
                  filt_probs: torch.Tensor) -> float:
    """KL(orig || filt): how different filtered is from original."""
    return float(F.kl_div(filt_probs.log(), orig_probs, reduction="sum"))


def detect_median(model, image_tensor: torch.Tensor,
                  kernel_size: int = 3,
                  kl_threshold: float = 0.5,
                  require_class_change: bool = False) -> dict:
    """Run median filter defense on a single (3, H, W) image in [0, 1].

    Args:
        model:               CIFAR-10 model (eval mode)
        image_tensor:        (3, H, W) float tensor in [0, 1]
        kernel_size:         median filter kernel (3 for 32×32 images)
        kl_threshold:        above this → ATTACK
        require_class_change: if True, only flag when class changes

    Returns dict:
        is_attack          bool
        kl_divergence      float
        class_changed      bool
        original_class     int
        filtered_class     int
        original_conf      float
        filtered_conf      float
        filtered_tensor    (3, H, W) tensor
    """
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)

    # Original prediction
    norm_orig  = ((image_tensor - m) / s).unsqueeze(0)
    with torch.no_grad():
        orig_probs = torch.softmax(model(norm_orig), dim=1)[0].cpu()
    orig_class = int(orig_probs.argmax())
    orig_conf  = float(orig_probs.max())

    # Filtered prediction
    filtered       = apply_median_filter(image_tensor, kernel_size)
    norm_filt      = ((filtered - m) / s).unsqueeze(0)
    with torch.no_grad():
        filt_probs = torch.softmax(model(norm_filt), dim=1)[0].cpu()
    filt_class = int(filt_probs.argmax())
    filt_conf  = float(filt_probs.max())

    kl            = kl_divergence(orig_probs, filt_probs)
    class_changed = orig_class != filt_class

    if require_class_change:
        # Only attack if class changed AND KL is high
        is_attack = class_changed and (kl > kl_threshold)
    else:
        # Standard rule: class same + high KL = attack
        is_attack = (not class_changed) and (kl > kl_threshold)

    return {
        "is_attack":       is_attack,
        "kl_divergence":   kl,
        "class_changed":   class_changed,
        "original_class":  orig_class,
        "filtered_class":  filt_class,
        "original_conf":   orig_conf,
        "filtered_conf":   filt_conf,
        "filtered_tensor": filtered,
        "safe_class":      filt_class,
        "safe_confidence": filt_conf,
    }
