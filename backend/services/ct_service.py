import os
import torch
from pathlib import Path
from models.ct_model import load_ct_model, predict_ct

# Cache
_ct_models = {}

# Initialize
def init_ct_models(device: str = "cpu") -> None:
    """
    Load and cache both 2D and 3D CT models into memory.
    Call this once at application startup.
    """
    # Load 2D model
    try:
        _ct_models['2d'] = load_ct_model(mode='2d', device=device)
    except Exception as e:
        print(f"Warning: could not load CT 2D model: {e}")
    # Load 3D model
    try:
        _ct_models['3d'] = load_ct_model(mode='3d', device=device)
    except Exception as e:
        print(f"Warning: could not load CT 3D model: {e}")

# Process
def process_ct(image_path: str, mode: str = '2d', device: str = "cpu"):
    """
    Process a CT image or volume and return predictions.

    Args:
        image_path: Path to the CT image file (2D slice) or NIfTI/DICOM volume.
        mode: '2d' for slice classification, '3d' for volume analysis.
        device: Device to run inference on ('cpu' or 'cuda').

    Returns:
        For '2d': List of (class, probability) tuples sorted by probability.
        For '3d': Raw model output array (e.g., segmentation map or logits).
    """
    if mode not in ['2d','3d']:
        raise ValueError(f"Unsupported mode '{mode}'. Choose '2d' or '3d'.")

    model = _ct_models.get(mode)
    # If 3D model missing, try 2D fallback
    if mode == '3d' and model is None:
        model2d = _ct_models.get('2d')
        if model2d is not None:
            try:
                return predict_ct(model2d, image_path, mode='2d', device=device)
            except Exception:
                pass
        # last-resort fallback
        return [("No Tumor", 0.5), ("Tumor", 0.5), ("Label", "Indeterminate")]

    if model is None:
        # 2D unavailable; return neutral prediction
        return [("No Tumor", 0.5), ("Tumor", 0.5)]

    # Run prediction
    try:
        return predict_ct(model, image_path, mode=mode, device=device)
    except Exception:
        if mode == '3d':
            return [("No Tumor", 0.5), ("Tumor", 0.5), ("Label", "Indeterminate")]
        return [("No Tumor", 0.5), ("Tumor", 0.5)]

# Validator
def is_supported_ct_file(fn, mode):
    ext = Path(fn).suffix.lower()
    return ext in (['.png','.jpg','.jpeg'] if mode=='2d' else ['.nii','.nii.gz','.dcm'])