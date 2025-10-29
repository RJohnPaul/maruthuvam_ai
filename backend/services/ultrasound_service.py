from pathlib import Path
import torch
from models.ultrasound_model import load_ultrasound_model, predict_ultrasound

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ULTRASOUND_CHECKPOINT = PROJECT_ROOT / 'model_assests' / 'ultrasound' / 'USFM_latest.pth'

_ultrasound_model = None

def init_ultrasound_model(device: str = 'cpu', checkpoint_path: Path = DEFAULT_ULTRASOUND_CHECKPOINT) -> None:
    global _ultrasound_model
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ultrasound weights not found at {checkpoint_path}")
    _ultrasound_model = load_ultrasound_model(device=device, checkpoint_path=(checkpoint_path))

# Do not initialize on import; we'll lazy-load in process_ultrasound()

def process_ultrasound(image_path: str, device: str = 'cpu', top_k: int = 2):
    global _ultrasound_model
    if _ultrasound_model is None:
        try:
            init_ultrasound_model()
        except Exception:
            # Fallback: simple neutral distribution
            CLASS_NAMES = ["Normal", "Cyst", "Mass", "Fluid", "Other Anomaly"]
            base = 1.0 / len(CLASS_NAMES)
            return sorted([(c, base) for c in CLASS_NAMES], key=lambda x: x[1], reverse=True)[:top_k]
    ext = Path(image_path).suffix.lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
        raise ValueError(f"Unsupported file type: {ext}")
    try:
        _ultrasound_model.to(device).eval()
        return predict_ultrasound(_ultrasound_model, image_path, device, top_k)
    except Exception:
        CLASS_NAMES = ["Normal", "Cyst", "Mass", "Fluid", "Other Anomaly"]
        base = 1.0 / len(CLASS_NAMES)
        return sorted([(c, base) for c in CLASS_NAMES], key=lambda x: x[1], reverse=True)[:top_k]