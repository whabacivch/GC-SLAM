"""
Descriptor Builder - Extraction and Composition.

Handles:
- Scan range histogram descriptors
- Image appearance descriptors (fixed-size histogram)
- Depth appearance descriptors (fixed-size histogram)
- Multi-modal descriptor composition
- Global NIG model maintenance

ALL mathematical operations call models.nig - NO math duplication.
"""

from typing import Optional, Tuple
import numpy as np
from sensor_msgs.msg import LaserScan

from fl_slam_poc.backend import NIGModel, NIG_PRIOR_KAPPA, NIG_PRIOR_ALPHA, NIG_PRIOR_BETA


class DescriptorBuilder:
    """
    Builds multi-modal descriptors from sensor data.
    
    Uses models.nig for all probabilistic descriptor operations (exact).
    """
    
    def __init__(self, descriptor_bins: int, depth_range_m: Tuple[float, float] = (0.1, 10.0)):
        """
        Args:
            descriptor_bins: Number of bins for scan descriptor histogram
            depth_range_m: (min_m, max_m) for depth histogram descriptor
        """
        self.descriptor_bins = descriptor_bins
        self.depth_range_m = (float(depth_range_m[0]), float(depth_range_m[1]))
        self.global_desc_model = None  # Global NIG model (uses models.nig)
    
    def scan_descriptor(self, msg: LaserScan) -> np.ndarray:
        """
        Extract scan range histogram descriptor.
        
        Returns:
            descriptor: np.ndarray of shape (descriptor_bins,)
        """
        ranges = np.asarray(msg.ranges, dtype=float).reshape(-1)
        valid = np.isfinite(ranges)
        valid &= (ranges >= float(msg.range_min))
        valid &= (ranges <= float(msg.range_max))
        
        if not np.any(valid):
            # Empty descriptor (all zeros)
            return np.zeros(self.descriptor_bins, dtype=float)
        
        r = ranges[valid]
        
        # Histogram of ranges
        hist, _ = np.histogram(r, bins=self.descriptor_bins, 
                               range=(float(msg.range_min), float(msg.range_max)))
        
        desc = np.asarray(hist, dtype=float)
        
        # Normalize to unit sum (probability distribution)
        desc_sum = float(np.sum(desc))
        if desc_sum > 1e-12:
            desc = desc / desc_sum
        
        return desc
    
    def image_descriptor(self, image_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract a fixed-size appearance descriptor from RGB image.

        Current implementation: grayscale intensity histogram (normalized).

        This is an exact, deterministic feature extraction step (no gating).
        """
        if image_data is None:
            return np.zeros(self.descriptor_bins, dtype=float)

        img = np.asarray(image_data)
        if img.ndim != 3 or img.shape[2] < 3:
            return np.zeros(self.descriptor_bins, dtype=float)

        # Grayscale conversion (RGB → luminance)
        img_f = img[..., :3].astype(np.float32)
        gray = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]

        hist, _ = np.histogram(gray.reshape(-1), bins=self.descriptor_bins, range=(0.0, 256.0))
        desc = np.asarray(hist, dtype=float)
        desc_sum = float(np.sum(desc))
        if desc_sum > 1e-12:
            desc = desc / desc_sum
        return desc
    
    def depth_descriptor(self, depth_m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract a fixed-size appearance descriptor from a depth image.

        Current implementation: depth value histogram over a fixed metric range.

        This is an exact, deterministic feature extraction step (no gating).
        """
        if depth_m is None:
            return np.zeros(self.descriptor_bins, dtype=float)

        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2:
            return np.zeros(self.descriptor_bins, dtype=float)

        dmin, dmax = self.depth_range_m
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            return np.zeros(self.descriptor_bins, dtype=float)

        vals = depth.reshape(-1)
        valid = np.isfinite(vals) & (vals >= dmin) & (vals <= dmax)
        if not np.any(valid):
            return np.zeros(self.descriptor_bins, dtype=float)

        hist, _ = np.histogram(vals[valid], bins=self.descriptor_bins, range=(dmin, dmax))
        desc = np.asarray(hist, dtype=float)
        desc_sum = float(np.sum(desc))
        if desc_sum > 1e-12:
            desc = desc / desc_sum
        return desc
    
    def compose_descriptor(self, 
                          scan_desc: np.ndarray,
                          image_feat: Optional[np.ndarray],
                          depth_feat: Optional[np.ndarray]) -> np.ndarray:
        """
        Compose multi-modal descriptor by concatenation.
        
        Returns:
            descriptor: Concatenated descriptor vector
        """
        # CRITICAL: Descriptor dimensionality must be FIXED over time.
        # If a modality is missing at a given timestep (e.g., depth TF not ready),
        # we insert a defined zero-vector for that modality to keep model shapes consistent.
        scan = np.asarray(scan_desc, dtype=float).reshape(-1)
        if scan.shape[0] != self.descriptor_bins:
            raise ValueError(f"scan_desc must have length {self.descriptor_bins}, got {scan.shape[0]}")

        image = np.zeros(self.descriptor_bins, dtype=float) if image_feat is None else np.asarray(image_feat, dtype=float).reshape(-1)
        if image.shape[0] != self.descriptor_bins:
            raise ValueError(f"image descriptor must have length {self.descriptor_bins}, got {image.shape[0]}")

        depth = np.zeros(self.descriptor_bins, dtype=float) if depth_feat is None else np.asarray(depth_feat, dtype=float).reshape(-1)
        if depth.shape[0] != self.descriptor_bins:
            raise ValueError(f"depth descriptor must have length {self.descriptor_bins}, got {depth.shape[0]}")

        return np.concatenate([scan, image, depth], axis=0)
    
    def init_global_model(self, descriptor: np.ndarray):
        """
        Initialize global NIG descriptor model.
        
        Uses models.nig (exact generative model).
        """
        if self.global_desc_model is None:
            self.global_desc_model = NIGModel.from_prior(
                mu=descriptor,
                kappa=NIG_PRIOR_KAPPA,
                alpha=NIG_PRIOR_ALPHA,
                beta=NIG_PRIOR_BETA
            )
    
    def update_global_model(self, descriptor: np.ndarray, weight: float):
        """
        Update global NIG model with new observation.
        
        Uses models.nig.update (exact Bayesian update).
        """
        if self.global_desc_model is not None:
            self.global_desc_model.update(descriptor, weight=weight)
    
    def get_global_model(self) -> Optional[NIGModel]:
        """Get current global NIG model."""
        return self.global_desc_model
    
    def copy_global_model(self) -> Optional[NIGModel]:
        """Get copy of global NIG model for new anchor."""
        if self.global_desc_model is None:
            return None
        return self.global_desc_model.copy()
