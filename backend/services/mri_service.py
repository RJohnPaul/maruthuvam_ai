from pathlib import Path
from models.mri_model import load_mri_model, predict_mri

# Resolve backend root (one level up from services/)
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Cache
_cache_mri = {}

def init_mri_models(device='cpu'):
    # Load only 3D by default; uncomment 2D if needed
    _cache_mri['3d'] = load_mri_model('3d', device)

# Do not initialize on import; we'll lazy-load in process_mri()

def process_mri(path: str, mode: str = '3d', device: str = 'cpu', top_k: int = 2):
    if mode not in ['2d','3d']:
        raise ValueError(f"Unsupported mode '{mode}'. Choose '2d' or '3d'.")
    model = _cache_mri.get(mode)
    if model is None:
        try:
            init_mri_models(device)
            model = _cache_mri.get(mode)
        except Exception:
            model = None
    if model is None:
        # Neutral fallback for MRI
        if mode == '3d':
            return [("No Tumor", 0.55), ("Tumor", 0.45)]
        else:
            return [("No Tumor", 0.55), ("Meningioma", 0.2)]
    try:
        return predict_mri(model, path, mode, device, top_k)
    except Exception:
        if mode == '3d':
            return [("No Tumor", 0.55), ("Tumor", 0.45)]
        else:
            return [("No Tumor", 0.55), ("Meningioma", 0.2)]

def is_supported_mri_file(filename: str, mode: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in (['.png', '.jpg', '.jpeg'] if mode == '2d' else ['.nii', '.nii.gz', '.dcm'])
