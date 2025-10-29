# MaruthuvamAI — Medical Imaging Assistant
A full‑stack assistant for medical image triage and report generation. Upload an image (X‑ray, CT, MRI, Ultrasound), get predictions with safety checks, and generate detailed reports using Gemini with clinician‑friendly guidance and follow‑ups.

## Highlights
- Multi‑modality uploads: X‑ray, CT (2D/3D), MRI (3D), Ultrasound
- Detailed AI reports powered by Gemini (configurable model; offline fallback)
- Map‑based doctor search and a simple landing‑page chatbot
- Strong safety rails:
	- Input guards (reject non‑chest X‑rays and non‑ultrasound images)
	- Confidence gating (low‑confidence results are marked Indeterminate)
	- Graceful fallbacks if local weights are missing
- X‑ray accuracy upgrade via TorchXRayVision pretrained models (auto‑enabled)

## Monorepo layout
```
Medical-Assistant-1/
├── frontend/                 # React + Vite + Tailwind
│   ├── src/                  # Pages, components, UI kit
│   ├── vite.config.js        # Vite + React plugin
│   └── .npmrc                # Local npm cache (avoids EACCES)
└── backend/                  # FastAPI + inference
		├── main.py               # API routes + report generation
		├── services/             # Modality services + guards + downloader
		│   ├── xray_service.py
		│   ├── ct_service.py
		│   ├── mri_service.py
		│   ├── ultrasound_service.py
		│   ├── image_guard.py    # Input validators (safety rails)
		│   └── model_downloader.py
		├── models/               # Lightweight model wrappers
		│   ├── xray_model.py     # + TorchXRayVision integration
		│   ├── ct_model.py
		│   ├── mri_model.py
		│   └── ultrasound_model.py
		├── model_assests/        # (Optional) local weights
		├── model_cache/          # (Auto) downloaded weights cache
		├── requirements.txt
		└── .env                  # Backend secrets (GEMINI_API_KEY, …)
```

## System requirements
- Backend: Python 3.10+ (3.11 recommended), macOS/Linux
- Frontend: Node.js 18+ and npm 9+
- GPU optional; CPU supported (slower)

## Quick start
### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Set your Gemini key (or put it in backend/.env)
export GEMINI_API_KEY=your_key_here
python3 -m uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- Dev URLs:
	- API: <http://127.0.0.1:8000>
	- Web: <http://localhost:5173>

## Environment variables (backend/.env)
- GEMINI_API_KEY=... (required)
- GEMINI_MODEL=models/gemini-2.0-pro-exp-02-05 (optional; new SDK)
- GEMINI_MODEL_LEGACY=gemini-1.5-pro (optional; legacy SDK)

## Safety and scope
- X‑ray guard rejects non‑chest images (e.g., hands) to avoid misleading diagnoses.
- Ultrasound guard checks grayscale + speckle‑like texture.
- Confidence gating marks low scores as Indeterminate and adds a cautionary note.
- If valid weights are not present, services return neutral fallbacks instead of crashing or over‑claiming.
- All reports include a disclaimer and are for preliminary guidance only; they do not replace professional medical advice.

## Accuracy: models and weights
- X‑ray: uses TorchXRayVision (DenseNet121, res224, multi‑dataset) automatically when local CheXNet weights are absent. This significantly improves chest finding signals compared to broken/unknown checkpoints.
- CT/MRI/Ultrasound: loaders support local weights; if you supply valid .pth/.pt/.pth.tar files at the paths below, they’ll be used immediately:
	- X‑ray (local optional): `backend/model_assests/xray/xray.pth.tar`
	- CT 2D: `backend/model_assests/ct/2d/ResNet50.pt`
	- CT 3D: `backend/model_assests/ct/3d/resnet_200.pth`
	- MRI 3D: `backend/model_assests/mri/3d/resnet_200.pth`
	- Ultrasound: `backend/model_assests/ultrasound/USFM_latest.pth`
- Downloader infra: `services/model_downloader.py` is ready for Hugging Face sources. We can add a simple YAML to map modalities → repo/files and auto‑download at first use on your approval.

## API reference (selected)
- POST `/predict/xray/` (multipart form)
	- file: image/jpeg|png|bmp
	- Returns: `{ predictions: [ [label, prob], ... ], note?: string }`
	- Safety: rejects non‑chest images; low confidence adds `note`.
- POST `/predict/ultrasound/` (multipart form)
	- file: image/jpeg|png|bmp
	- Returns: `{ symptoms: [...], disease: string, report: string, note?: string }`
	- Safety: rejects non‑ultrasound images; low confidence adds `note`.
- POST `/predict/ct/2d/`
	- file: image/jpeg|png|bmp
	- Returns: `{ symptoms, disease, report }`
- POST `/predict/ct/3d/`
	- file: 3D NIfTI preferred; gracefully falls back for 2D
	- Returns: summary with “Condition Detected” parsing
- POST `/predict/mri/3d/`
	- file: 3D NIfTI preferred; gracefully falls back for 2D
	- Returns: summary with “Condition Detected” parsing
- POST `/generate-report/{modality}/`
	- Generates a detailed report using Gemini (multimodal when image bytes + mime are given)

## Frontend notes
- The project uses a local npm cache (`frontend/.npmrc`) to prevent EACCES errors during install on macOS/Linux.
- UI: Tailwind + shadcn/ui components; PDF generation via @react-pdf/renderer; interactive components under `src/components`.

## Troubleshooting
- npm install permission errors: already mitigated by local cache; if needed `rm -rf frontend/.npm-cache && npm install`.
- “invalid load key, 'v'” on backend: indicates invalid/unknown PyTorch checkpoint. Supply a valid `.pth/.pt/.pth.tar` or use the built‑in XRV backend for X‑ray (already active). Other modalities return safe fallbacks until valid weights are provided.
- Gemini errors / empty responses: the backend now has an offline fallback that still returns a safe textual summary. Ensure GEMINI_API_KEY is set for best results.
- Large bundle warning (frontend build): consider route‑level code splitting or Vite `manualChunks`.

## Roadmap (optional)
- Add `model_sources.yaml` with HF repo/file mappings and auto‑download at first use
- Implement `/health/models` endpoint to report which models are loaded vs fallback
- Add tests for guards, confidence gating, and report formatting
- Expand ultrasound and CT/MRI models with public checkpoints (as licenses permit)

## License
MIT © 2025

## Credits
- Core team: Sumit Singh, Somil Gupta, Abhishek

# maruthuvam_ai
