"""EXPA Anomaly Detector — 4-metric vote-based detection (standalone).

Metrics (each produces a flag):
    1. Spatial Entropy    — low  → concentrated attention  → suspicious
    2. Peak-to-Mean Ratio — high → sharp localized peak    → suspicious
    3. Top-k Energy Ratio — high → energy in few pixels    → suspicious
    4. Cross-Layer Corr.  — low  → inconsistent layers     → suspicious
       KEY IDEA: natural objects create consistent attention across early
       and late layers; adversarial patches hijack only deep layers.

Decision (vote mode):
    2 or more flags triggered → ATTACK

All thresholds are tuned for CIFAR-10 ResNet-20:
    - late layer (layer3): 64×8×8  → max entropy ≈ log2(64) = 6.0 bits
    - early layer (layer1): 16×32×32
"""

import torch
import torch.nn.functional as F


# ── Default thresholds (CIFAR-10 ResNet-20) ───────────────────────────────────
DEFAULT_CONFIG = {
    "entropy_threshold":    5.0,    # below  = concentrated = suspicious
    "peak_mean_threshold":  3.5,    # above  = sharp peak   = suspicious
    "topk_ratio_threshold": 0.25,   # above  = few pixels   = suspicious
    "topk_percent":         0.05,   # top-k uses top 5% of positions
    "cross_layer_threshold": 0.5,   # below  = inconsistent = suspicious
    "mask_confidence_drop":  0.15,  # min drop to confirm adversarial hotspot
    "detection_mode":        "vote",
}


class AnomalyDetector:
    """Detects adversarial patches using EigenCAM heatmap analysis."""

    def __init__(self, config: dict = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.entropy_thr    = cfg["entropy_threshold"]
        self.peak_mean_thr  = cfg["peak_mean_threshold"]
        self.topk_ratio_thr = cfg["topk_ratio_threshold"]
        self.topk_percent   = cfg["topk_percent"]
        self.cross_layer_thr = cfg["cross_layer_threshold"]
        self.mode           = cfg["detection_mode"]

    # ── Individual metrics ────────────────────────────────────────────────────

    def spatial_entropy(self, heatmap: torch.Tensor) -> float:
        """Low entropy = concentrated attention = suspicious."""
        flat  = heatmap.flatten().float()
        total = flat.sum()
        if total < 1e-8:
            return 0.0
        p       = flat / total
        p       = p[p > 1e-10]
        entropy = -(p * torch.log2(p)).sum()
        return float(entropy)

    def peak_to_mean_ratio(self, heatmap: torch.Tensor) -> float:
        """High ratio = sharp localized peak = suspicious."""
        mean_val = heatmap.mean()
        if mean_val < 1e-8:
            return 0.0
        return float(heatmap.max() / mean_val)

    def topk_energy_ratio(self, heatmap: torch.Tensor) -> float:
        """High ratio = energy concentrated in few pixels = suspicious."""
        flat         = heatmap.flatten()
        total_energy = flat.sum()
        if total_energy < 1e-8:
            return 0.0
        k           = max(1, int(len(flat) * self.topk_percent))
        topk_values, _ = torch.topk(flat, k)
        return float(topk_values.sum() / total_energy)

    def cross_layer_correlation(self, heatmap_early: torch.Tensor,
                                heatmap_late: torch.Tensor) -> float:
        """Pearson correlation between early and late heatmaps.

        Natural objects: both layers focus on the same region → high corr.
        Adversarial patches: late layer hijacked, early layer not → low corr.
        """
        h = max(heatmap_early.shape[0], heatmap_late.shape[0])
        w = max(heatmap_early.shape[1], heatmap_late.shape[1])

        early = F.interpolate(
            heatmap_early.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode="bilinear", align_corners=False
        ).flatten()

        late = F.interpolate(
            heatmap_late.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode="bilinear", align_corners=False
        ).flatten()

        e_c = early - early.mean()
        l_c = late  - late.mean()
        num = (e_c * l_c).sum()
        den = e_c.norm() * l_c.norm()

        return float(num / den) if den > 1e-8 else 0.0

    # ── Main interface ────────────────────────────────────────────────────────

    def compute_scores(self, heatmap_late: torch.Tensor,
                       heatmap_early: torch.Tensor = None) -> dict:
        entropy   = self.spatial_entropy(heatmap_late)
        peak_mean = self.peak_to_mean_ratio(heatmap_late)
        topk      = self.topk_energy_ratio(heatmap_late)

        scores = {
            "entropy":          entropy,
            "peak_mean":        peak_mean,
            "topk_energy":      topk,
            "entropy_flag":     entropy   < self.entropy_thr,
            "peak_mean_flag":   peak_mean > self.peak_mean_thr,
            "topk_energy_flag": topk      > self.topk_ratio_thr,
        }

        if heatmap_early is not None:
            cross = self.cross_layer_correlation(heatmap_early, heatmap_late)
            scores["cross_layer_corr"] = cross
            scores["cross_layer_flag"] = cross < self.cross_layer_thr
        else:
            scores["cross_layer_corr"] = None
            scores["cross_layer_flag"] = False

        return scores

    def detect(self, heatmap_late: torch.Tensor,
               heatmap_early: torch.Tensor = None) -> tuple:
        """Decide if a heatmap indicates an adversarial patch.

        Returns:
            is_adversarial: bool
            scores: dict with all metric values and flags
        """
        scores = self.compute_scores(heatmap_late, heatmap_early)

        flags = [
            scores["entropy_flag"],
            scores["peak_mean_flag"],
            scores["topk_energy_flag"],
        ]
        if scores["cross_layer_corr"] is not None:
            flags.append(scores["cross_layer_flag"])

        if self.mode == "vote":
            is_adversarial = sum(flags) >= 2
        elif self.mode == "any":
            is_adversarial = any(flags)
        else:
            is_adversarial = sum(flags) >= 2

        scores["num_flags"] = sum(flags)
        return is_adversarial, scores

    def detect_batch(self, heatmaps_late: torch.Tensor,
                     heatmaps_early: torch.Tensor = None) -> list:
        """Detect for a batch. Returns list of (is_adversarial, scores)."""
        results = []
        for i in range(heatmaps_late.shape[0]):
            early_i = heatmaps_early[i] if heatmaps_early is not None else None
            results.append(self.detect(heatmaps_late[i], early_i))
        return results
