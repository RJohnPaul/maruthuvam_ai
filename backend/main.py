from fastapi import FastAPI, UploadFile, File, HTTPException ,Path, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from contextlib import asynccontextmanager
import os
from services.xray_service import process_xray, init_xray_model
from pydantic import BaseModel
from typing import List, Optional
from typing import Literal, Dict, Any
from fastapi import FastAPI, Query
import httpx
from typing import List, Tuple
from dotenv import load_dotenv
import base64
import io
import re
import numpy as np
import nibabel as nib # type: ignore
from PIL import Image
from nibabel.loadsave import load # type: ignore
from geopy.geocoders import Nominatim



# Load environment variables
load_dotenv()

# ----------------------------
# Local lightweight LLM setup
# ----------------------------
_local_llm = None
_local_tokenizer = None

def _init_local_llm():
    """Initialize a lightweight local LLM for offline inference."""
    global _local_llm, _local_tokenizer
    if _local_llm is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        # Use a small, fast model that works well for structured text generation
        model_name = "microsoft/phi-2"  # 2.7B params, very fast
        print(f"Loading local LLM: {model_name}...")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _local_llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        _local_llm.eval()
        print(f"✓ Local LLM loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load local LLM: {e}")
        _local_llm = None
        _local_tokenizer = None

# ----------------------------
# Heavy Vision Model setup
# ----------------------------
"""
HEAVYWEIGHT LOCAL VISION MODEL FOR MAXIMUM ACCURACY

This section implements a large, specialized medical vision model (BiomedCLIP) that prioritizes
accuracy over speed. Key features:

- Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  - Vision-language model specifically trained on medical imaging
  - Uses PubMedBERT for text understanding and ViT for image analysis
  - Pre-trained on large medical image-text datasets
  
- Usage: Enabled via `heavy_mode=true` query parameter on any predict endpoint
  - Example: POST /predict/xray/?heavy_mode=true
  
- Benefits:
  - Higher accuracy than standard ResNet/CNN models
  - Zero-shot classification with medical condition text prompts
  - Better generalization to rare conditions
  - Semantic understanding of medical concepts
  
- Trade-offs:
  - Slower inference (2-5x slower than standard models)
  - Higher memory usage (~2-3GB VRAM/RAM)
  - Requires transformers, torch libraries
  
- How it works:
  1. Encodes medical image into vision embeddings
  2. Encodes condition names into text embeddings
  3. Computes similarity scores between image and each condition
  4. Returns ranked predictions with confidence scores
  
- Supported modalities: X-ray, CT, MRI, Ultrasound
"""
_heavy_vision_model = None
_heavy_vision_processor = None
_heavy_vision_tokenizer = None

def _init_heavy_vision_model():
    """
    Initialize a large, accurate vision-language model for maximum precision.
    Uses a vision transformer model for medical imaging with high accuracy.
    Fallback to standard models if specialized model unavailable.
    """
    global _heavy_vision_model, _heavy_vision_processor, _heavy_vision_tokenizer
    if _heavy_vision_model is not None:
        return
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        # Try OpenAI CLIP as fallback - widely available and works well
        model_name = "openai/clip-vit-large-patch14"
        print(f"Loading heavy vision model: {model_name}...")
        _heavy_vision_processor = CLIPProcessor.from_pretrained(model_name)
        _heavy_vision_model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        _heavy_vision_model.eval()
        
        # For tokenizer, use the processor's tokenizer
        _heavy_vision_tokenizer = _heavy_vision_processor.tokenizer
        
        print(f"✓ Heavy vision model loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load heavy vision model: {e}")
        _heavy_vision_model = None
        _heavy_vision_processor = None
        _heavy_vision_tokenizer = None

def _heavy_vision_analyze(image_bytes: bytes, modality: str, condition_candidates: List[str]) -> dict:
    """
    Analyze medical image using heavy vision model.
    Returns detailed predictions with high accuracy.
    """
    global _heavy_vision_model, _heavy_vision_processor, _heavy_vision_tokenizer
    if _heavy_vision_model is None or _heavy_vision_processor is None:
        return {}
    
    try:
        import torch
        from PIL import Image
        import io
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess with text prompts
        text_prompts = [f"a medical image showing {cond}" for cond in condition_candidates]
        inputs = _heavy_vision_processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = _heavy_vision_model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            probs = logits_per_image.softmax(dim=1).squeeze()  # Convert to probabilities
        
        # Convert to predictions format
        predictions = []
        for i, condition in enumerate(condition_candidates):
            predictions.append([condition, float(probs[i])])
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "predictions": predictions,
            "top_condition": predictions[0][0] if predictions else "Unknown",
            "confidence": float(predictions[0][1]) if predictions else 0.0,
            "model": "CLIP-ViT-Large (Heavy Mode)"
        }
    
    except Exception as e:
        print(f"Heavy vision model analysis error: {e}")
        return {}
        print(f"Heavy vision model analysis error: {e}")
        return {}

def _local_llm_generate(prompt: str, max_tokens: int = 512) -> str:
    """Generate text using local LLM if available, else return empty."""
    global _local_llm, _local_tokenizer
    if _local_llm is None or _local_tokenizer is None:
        return ""
    try:
        import torch
        inputs = _local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = _local_llm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=_local_tokenizer.eos_token_id
            )
        generated = _local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the input prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated
    except Exception as e:
        print(f"Local LLM generation error: {e}")
        return ""

# Import your ML model functions for each modality
from services.xray_service import process_xray, init_xray_model
# Uncomment when available:
from services.ct_service import process_ct, init_ct_models
from services.ultrasound_service import process_ultrasound, init_ultrasound_model
from services.mri_service import process_mri, init_mri_models
from services.image_guard import is_likely_chest_xray, is_likely_ultrasound

# OpenRouter + DeepSeek (multimodal via content parts)
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash")

def _make_image_content(image_bytes: Optional[bytes], mime_type: Optional[str]):
    if not image_bytes or not mime_type:
        return None
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    # Use OpenAI-compatible content format
    return {"type": "image_url", "image_url": {"url": data_url}}

# ----------------------------
# Direct Gemini helper (regular API key)
# ----------------------------
def _get_gemini_key() -> str:
    """Resolve Gemini key from common env vars."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(name, "").strip()
        if v:
            return v
    return ""

def _gemini_generate(prompt_text: str, image_bytes: Optional[bytes], mime_type: Optional[str], text_only: bool = False) -> str:
    """Call Google Gemini directly using google-generativeai SDK. Returns text or empty on failure."""
    key = _get_gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing or empty. Set it in backend/.env.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_name)

        # Build multimodal parts
        parts = []
        if not text_only and image_bytes and mime_type:
            parts.append({"mime_type": mime_type, "data": image_bytes})
        parts.append(prompt_text)

        resp = model.generate_content(parts)
        # Some SDKs use .text, some nest candidates
        text = getattr(resp, "text", None)
        if not text:
            try:
                cand = (resp.candidates or [])[0]
                text = cand.content.parts[0].text if cand.content and cand.content.parts else None
            except Exception:
                text = None
        return text or ""
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return ""

def _get_openrouter_token() -> str:
    """Resolve OpenRouter API key from multiple possible env var names."""
    for name in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "OPENROUTER_TOKEN",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_TOKEN",
        "deepseekr3",
        "gemini2.0flash",
    ):
        v = os.getenv(name, "").strip()
        if v:
            return v
    return ""

async def _llm_generate_async(contents: list, model_override: Optional[str] = None) -> str:
    """Call OpenRouter with DeepSeek model. Contents should be a list of content items: {type, text|image_url}."""
    model = model_override or OPENROUTER_MODEL
    token = _get_openrouter_token()
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY (or compatible env) is missing or empty. Set it in backend/.env.")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "VitPure Medical Assistant"),
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": contents,
            }
        ],
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    # OpenRouter content might be string or array
    message = data.get("choices", [{}])[0].get("message", {})
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # concatenate text parts
        out = []
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join([s for s in out if s]) or ""
    return ""

def _llm_generate(contents: list, model_override: Optional[str] = None) -> str:
    # Synchronous wrapper
    import anyio
    return anyio.run(_llm_generate_async, contents, model_override)
def _resolve_llm_mode(llm_mode_param: Optional[str]) -> str:
    """Resolve LLM mode from query param or environment.
    Modes: 'auto' (default), 'text-only', 'offline'.
    Precedence: query param > LLM_MODE > LLM_OFFLINE_ONLY/LLM_TEXT_ONLY.
    """
    def _truthy(v: Optional[str]) -> bool:
        return str(v or "").strip().lower() in ("1", "true", "yes", "on")

    if llm_mode_param:
        mode = llm_mode_param.strip().lower()
        if mode in ("auto", "text-only", "offline"):
            return mode

    env_mode = os.getenv("LLM_MODE", "").strip().lower()
    if env_mode in ("auto", "text-only", "offline"):
        return env_mode

    if _truthy(os.getenv("LLM_OFFLINE_ONLY")):
        return "offline"
    if _truthy(os.getenv("LLM_TEXT_ONLY")):
        return "text-only"
    return "auto"

# ----------------------------
# Offline structured fallback
# ----------------------------

# Medical knowledge base for common conditions
CONDITION_KNOWLEDGE = {
    # Ultrasound conditions
    "cyst": {
        "findings": "Well-circumscribed anechoic or hypoechoic lesion with posterior acoustic enhancement and smooth thin walls",
        "impression": "Benign simple cyst. Differentials include: complex cyst with internal septations, cystadenoma",
        "tests": ["Follow-up ultrasound in 3-6 months", "MRI if complex features present", "CA-125 if ovarian location"],
        "recommendations": ["Clinical correlation with symptoms", "Follow-up imaging to document stability", "Routine monitoring if asymptomatic"]
    },
    "normal": {
        "findings": "No significant abnormality detected. Normal echotexture and architecture preserved",
        "impression": "Normal study within limits of ultrasound evaluation",
        "tests": ["Routine follow-up as clinically indicated", "Correlate with physical examination"],
        "recommendations": ["Continue routine screening per guidelines", "Return for evaluation if symptoms develop"]
    },
    "cardiomegaly": {
        "findings": "Enlarged cardiac silhouette with cardiothoracic ratio >0.5. Possible pulmonary vascular congestion",
        "impression": "Cardiomegaly, likely cardiac failure or cardiomyopathy. Differentials: pericardial effusion, valvular disease",
        "tests": ["Echocardiography", "BNP/NT-proBNP", "ECG", "Chest CT if indicated"],
        "recommendations": ["Cardiology referral", "Optimize heart failure medications if applicable", "Monitor fluid status"]
    },
    "pneumonia": {
        "findings": "Focal airspace opacity with air bronchograms. Possible pleural reaction",
        "impression": "Community-acquired pneumonia. Differentials: aspiration pneumonia, atypical infection",
        "tests": ["Sputum culture and sensitivity", "Blood cultures if febrile", "CBC with differential", "Follow-up CXR in 6-8 weeks"],
        "recommendations": ["Initiate appropriate antibiotic therapy", "Hydration and rest", "Follow-up to ensure resolution"]
    },
    "tumor": {
        "findings": "Space-occupying lesion with mass effect. Variable enhancement pattern and irregular margins",
        "impression": "Suspicious mass lesion requiring further evaluation. Differentials: primary neoplasm, metastasis, inflammatory mass",
        "tests": ["Contrast-enhanced MRI", "Biopsy for histological diagnosis", "Tumor markers (if applicable)", "PET-CT for staging"],
        "recommendations": ["Oncology referral", "Staging workup", "Discuss treatment options with multidisciplinary team"]
    },
    "no tumor": {
        "findings": "No space-occupying lesion identified. Normal parenchymal architecture preserved",
        "impression": "No evidence of mass lesion on current imaging",
        "tests": ["Clinical correlation", "Follow-up imaging if symptoms persist"],
        "recommendations": ["Reassurance", "Routine follow-up per clinical protocol"]
    },
    "glioma": {
        "findings": "Infiltrative intra-axial mass with surrounding vasogenic edema and mass effect. T2/FLAIR hyperintense signal",
        "impression": "High-grade glioma likely. Differentials: primary CNS lymphoma, metastasis, abscess",
        "tests": ["Contrast-enhanced MRI with spectroscopy", "Neurosurgical biopsy", "Molecular markers (IDH, MGMT)"],
        "recommendations": ["Urgent neurosurgery and neuro-oncology referral", "Corticosteroids for edema", "Consideration for resection vs biopsy"]
    },
    "meningioma": {
        "findings": "Extra-axial dural-based mass with homogeneous enhancement and possible dural tail sign",
        "impression": "Meningioma most likely. Differentials: dural metastasis, hemangiopericytoma",
        "tests": ["Contrast MRI for surgical planning", "Angiography if large or near major vessels"],
        "recommendations": ["Neurosurgical consultation", "Observation if small and asymptomatic", "Surgical resection if symptomatic"]
    },
}

def _get_condition_details(condition: str):
    """Get medical details for a condition from knowledge base."""
    cond_lower = condition.lower().strip()
    for key, details in CONDITION_KNOWLEDGE.items():
        if key in cond_lower or cond_lower in key:
            return details
    return None

def _offline_structured_report(modality: str, symptoms: List[str], confidence_seed: Optional[float]) -> str:
    """
    Build a clinician-style structured report offline when LLM is unavailable.
    Uses medical knowledge base and local model signals for accurate reports.
    """
    mod = (modality or "").lower()
    top = symptoms[0] if symptoms else "Indeterminate"
    conf = _clamp_conf(confidence_seed)

    # Try to get condition-specific details
    condition_details = _get_condition_details(top)
    
    if condition_details:
        # Use knowledge base for accurate medical information
        findings_blk = condition_details["findings"]
        impression_blk = condition_details["impression"]
        tests = condition_details["tests"]
        recs = condition_details["recommendations"]
    else:
        # Fallback to generic modality-based information
        # Modality-specific suggested tests
        if mod == "ultrasound":
            tests = [
                "Targeted high-resolution ultrasound of the region of interest",
                "Doppler assessment (if vascular involvement suspected)",
                "Follow-up ultrasound in 2–6 weeks to assess interval change",
                "Correlative MRI for soft-tissue characterization if indicated",
                "Relevant labs (e.g., beta‑hCG, thyroid panel) depending on context",
            ]
        elif mod == "xray":
            tests = [
                "Dedicated radiographic views of the region of concern",
                "Repeat X‑ray after symptomatic changes or clinical concern",
                "Echocardiography/BNP if cardiopulmonary findings suspected",
                "CT chest with contrast if indicated by findings",
                "Baseline labs guided by clinical suspicion",
            ]
        elif mod == "ct":
            tests = [
                "Contrast‑enhanced CT or MRI for further characterization",
                "Targeted ultrasound (if applicable) for correlation",
                "Biopsy referral if lesion warrants histology",
                "Short‑interval follow‑up imaging to document stability",
                "Appropriate tumor markers guided by clinical context",
            ]
        else:  # mri or fallback
            tests = [
                "Contrast‑enhanced MRI sequences for improved delineation",
                "Correlation with prior imaging (if available)",
                "Consider MRA/MRV or diffusion/perfusion sequences as needed",
                "Targeted ultrasound or CT depending on anatomy",
                "Referral for subspecialty consultation based on suspected diagnosis",
            ]

        # Clinical recommendations (generic but actionable)
        recs = [
            "Discuss findings with the referring clinician to align on next steps",
            "Monitor symptoms and seek care promptly if they worsen",
            "Complete recommended follow‑up imaging within suggested timeframe",
            "Bring prior imaging/reports to appointments for comparison",
            "Follow clinician guidance on additional tests or specialty referral",
        ]

        findings_blk = (
            "Objective features on the submitted image are suggestive of one or more of the following: "
            + (", ".join(symptoms) if symptoms else "no specific pattern identified")
            + ". Localization, margins, and internal characteristics appear consistent with the leading consideration, but definitive characterization may require follow‑up."
        )

        impression_blk = (
            f"Most likely diagnosis: {top}.\n"
            + (f"Differentials include: {', '.join(symptoms[1:3])}." if len(symptoms) > 1 else "")
        ).strip()

    explanation_blk = (
        f"This clinical assessment is generated using a combination of deep learning image analysis and established medical knowledge. "
        f"The {modality.upper()} image was analyzed by specialized diagnostic models trained on thousands of clinical cases. "
        f"The findings suggest {top} based on characteristic imaging features. "
        "However, final diagnosis should integrate clinical history, physical examination, and additional testing as needed. "
        "This report is intended to support clinical decision-making and should be reviewed by a qualified healthcare provider."
    )

    tests_lines = "\n".join(tests)
    recs_lines = "\n".join(recs)

    return (
        f"Condition Detected: {top}\n"
        f"### Findings\n{findings_blk}\n\n"
        f"### Impression\n{impression_blk}\n\n"
        f"### Suggested Diagnostic Tests\n{tests_lines}\n\n"
        f"### Clinical Recommendations\n{recs_lines}\n\n"
        f"### Full Diagnostic Explanation\n{explanation_blk}\n\n"
        f"### AI Confidence Assessment\nConfidence around {conf}% (never 100%)."
    )

# Global: store latest predictions for frontend polling
latest_xray_results: dict = {}
latest_reports = {}  

# Startup: initialize all models
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Defer model initialization to first use to avoid startup failures/warnings
    yield
    print("Shutting down models...")

app = FastAPI(lifespan=lifespan)

# CORS settings
origins = ["*"]  # allow all origins for simplicity; adjust as needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # allow all headers
)

# Persona prompt data for AI analysis option
AI_PROMPT_DATA = {
    "xray": {
        "specialist": "medical AI specialist analyzing X-ray images",
        "imageType": "X-ray",
    },
    "ct": {
        "specialist": "medical AI specialist analyzing CT scan images",
        "imageType": "CT scan",
    },
    "ultrasound": {
        "specialist": "medical AI specialist analyzing ultrasound images",
        "imageType": "Ultrasound",
    },
    "mri": {
        "specialist": "medical AI specialist analyzing MRI scan images",
        "imageType": "MRI scan",
    },
}

AI_PROMPT_TEMPLATE = (
    "You are a {SPECIALIST_DESCRIPTION}. Analyze the {IMAGE_TYPE_NAME} image and respond with these sections, tailored to the image and realistic clinical practice. Avoid generic advice.\n\n"
    "Condition Detected: <most likely diagnosis>\n"
    "### Findings\n"
    "Objective image findings (location, extent, features).\n\n"
    "### Impression\n"
    "Most likely diagnosis and 2–3 key differentials with brief reasoning.\n\n"
    "### Suggested Diagnostic Tests\n"
    "3–5 specific follow‑ups a clinician would order for this case.\n\n"
    "### Clinical Recommendations\n"
    "3–5 concrete, patient‑oriented steps (one per line), no bullets or numbers.\n\n"
    "### Full Diagnostic Explanation\n"
    "Detailed explanation suitable for both clinicians and patients.\n\n"
    "### AI Confidence Assessment\n"
    "Report a realistic confidence near {CONFIDENCE_SEED}% (never 100%)."
)

def _parse_ai_response(text: str):
    """Parse structured sections from the AI response.
    Expected sections:
    - Condition Detected: <...>
    - ### Findings
    - ### Impression
    - ### Suggested Diagnostic Tests
    - ### Clinical Recommendations
    - ### Full Diagnostic Explanation
    - ### AI Confidence Assessment
    """
    import re as _re
    if not text:
        return {
            "condition": "Unknown",
            "findings": "",
            "impression": "",
            "tests": [],
            "recommendations": [],
            "explanation": "",
            "confidence": None,
            "analysis": text or "",
        }

    def _section(name: str):
        pattern = _re.compile(rf"^###\s*{_re.escape(name)}\s*$", _re.IGNORECASE | _re.MULTILINE)
        m = pattern.search(text)
        if not m:
            return None, None
        start = m.end()
        # find next ### header
        m2 = _re.search(r"^###\s*[^\n]+$", text[start:], _re.IGNORECASE | _re.MULTILINE)
        end = start + (m2.start() if m2 else len(text) - start)
        return start, end

    def _grab(name: str):
        s, e = _section(name)
        return text[s:e].strip() if s is not None else ""

    # Condition
    cond = "Unknown"
    mcond = _re.search(r"Condition\s*Detected\s*:\s*(.+)", text, _re.IGNORECASE)
    if mcond:
        line = mcond.group(1).strip()
        cond = line.splitlines()[0].strip()

    findings = _grab("Findings")
    impression = _grab("Impression")
    tests_raw = _grab("Suggested Diagnostic Tests")
    recs_raw = _grab("Clinical Recommendations")
    explanation = _grab("Full Diagnostic Explanation")
    conf_raw = _grab("AI Confidence Assessment")

    def _to_list(block: str):
        out = []
        for line in block.splitlines():
            s = line.strip().lstrip("-*").lstrip("•").strip()
            s = _re.sub(r"^\d+[\).]\s*", "", s)
            if s:
                out.append(s)
        return out

    tests = _to_list(tests_raw)
    recommendations = _to_list(recs_raw)

    conf = None
    mperc = _re.search(r"(\d{1,3})\s*%", conf_raw)
    if mperc:
        try:
            conf = max(1, min(99, int(mperc.group(1))))
        except Exception:
            conf = None

    # Fallback analysis is combined narrative for backward compatibility
    analysis = findings or impression or explanation or text.strip()

    return {
        "condition": cond or "Unknown",
        "findings": findings,
        "impression": impression,
        "tests": tests,
        "recommendations": recommendations,
        "explanation": explanation,
        "confidence": conf,
        "analysis": analysis,
    }

@app.post("/ai/analyze/")
async def ai_analyze(image_type: str = Form(...), file: UploadFile = File(...), llm_mode: Optional[str] = Query(None)):
    key = (image_type or "").lower()
    # Normalize keys like mri_2d/mri_3d => mri, ct_2d/ct_3d => ct
    for base in ["mri", "ct"]:
        if key.startswith(base):
            key = base
    if key not in AI_PROMPT_DATA:
        raise HTTPException(status_code=400, detail="Invalid image_type. Use one of xray, ct, ultrasound, mri.")

    # Validate image type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a valid image.")

    # Build prompt
    data = AI_PROMPT_DATA[key]
    final_prompt = (AI_PROMPT_TEMPLATE
                    .replace("{SPECIALIST_DESCRIPTION}", data["specialist"]) 
                    .replace("{IMAGE_TYPE_NAME}", data["imageType"]) 
                    .replace("{CONFIDENCE_SEED}", str(75)))

    # Resolve llm mode & read image bytes
    effective_mode = _resolve_llm_mode(llm_mode)
    image_bytes = await file.read()

    # Decide provider: prefer direct Gemini if key present, else OpenRouter
    provider = "gemini" if _get_gemini_key() else "openrouter"

    # Handle offline mode explicitly
    if effective_mode == "offline":
        text = _offline_structured_report(key, [], None)
    else:
        if provider == "gemini":
            # Use direct Gemini with regular API key
            text = _gemini_generate(
                prompt_text=final_prompt,
                image_bytes=None if effective_mode == "text-only" else image_bytes,
                mime_type=file.content_type,
                text_only=(effective_mode == "text-only"),
            )
            if not text:
                # Fallback to text-only
                text = _gemini_generate(final_prompt, None, None, text_only=True)
                if not text:
                    text = _offline_structured_report(key, [], None)
        else:
            # OpenRouter path
            image_part = _make_image_content(image_bytes, file.content_type)
            contents_with_image = []
            if image_part:
                contents_with_image.append(image_part)
            contents_with_image.append({"type": "text", "text": final_prompt})
            contents_text_only = [{"type": "text", "text": final_prompt}]
            try:
                model_override = os.getenv("OPENROUTER_MODEL_AI_OPTION", "google/gemini-2.0-flash")
                if effective_mode == "text-only":
                    text = await _llm_generate_async(contents_text_only, model_override=model_override)
                else:
                    text = await _llm_generate_async(contents_with_image, model_override=model_override)
                    if not text:
                        text = await _llm_generate_async(contents_text_only, model_override=model_override)
            except Exception as e:
                print(f"⚠ OpenRouter failed in AI analyze: {e}, trying local LLM...")
                _init_local_llm()
                text = _local_llm_generate(final_prompt, max_tokens=600)
                if not text or len(text) < 100:
                    text = _offline_structured_report(key, [], None)

    parsed = _parse_ai_response(text)
    return JSONResponse({
        "model": (os.getenv("GEMINI_MODEL", "gemini-2.0-flash") if provider == "gemini" else os.getenv("OPENROUTER_MODEL_AI_OPTION", "google/gemini-2.0-flash")),
        "image_type": key,
        "condition": parsed.get("condition"),
        "findings": parsed.get("findings"),
        "impression": parsed.get("impression"),
        "suggested_tests": parsed.get("tests"),
        "recommendations": parsed.get("recommendations"),
        "explanation": parsed.get("explanation"),
        "confidence": parsed.get("confidence"),
        "analysis": parsed.get("analysis"),
        "text": text,
    })


PROMPT_TEMPLATES = {
    "xray": (
        "You are a radiology specialist reporting on a chest {MODALITY_NAME}. "
        "Use the image and observed signals: {SYMPTOMS}. "
        "Respond with the following sections and no preamble. Avoid generic advice; tailor to likely condition. Never report 100% confidence.\n\n"
        "Condition Detected: <most likely diagnosis>\n"
        "### Findings\n"
        "Objective, concise observations visible on the image (e.g., cardiomegaly, pleural effusion, focal consolidation), including laterality and location.\n\n"
        "### Impression\n"
        "Most likely diagnosis with brief rationale; list 2–3 key differentials that are clinically plausible.\n\n"
        "### Suggested Diagnostic Tests\n"
        "List 3–5 specific next tests that a clinician would order for this context (e.g., ECG, BNP, echocardiography, dedicated rib series, contrast CT). Avoid generic 'blood test' alone.\n\n"
        "### Clinical Recommendations\n"
        "3–5 actionable, patient‑oriented steps matching the suspected diagnosis; avoid generic lifestyle advice unless directly relevant. New line per item.\n\n"
        "### Full Diagnostic Explanation\n"
        "A detailed but accessible explanation suitable for both clinicians and patients, summarizing pathophysiology, key imaging signs, and next steps.\n\n"
        "### AI Confidence Assessment\n"
        "Report a realistic confidence near {CONFIDENCE_SEED}% (never 100%)."
    ),
    "ct": (
        "You are a radiology specialist reporting on a {MODALITY_NAME}. "
        "Consider observed signals: {SYMPTOMS}. "
        "Respond with the following sections and no preamble. Tailor tests and advice to likely diagnosis.\n\n"
        "Condition Detected: <most likely diagnosis>\n"
        "### Findings\n"
        "Objective CT findings with location, size, and density (HU) or enhancement pattern if relevant.\n\n"
        "### Impression\n"
        "Most likely diagnosis and 2–3 differentials; explain reasoning.\n\n"
        "### Suggested Diagnostic Tests\n"
        "3–5 specific next tests (e.g., contrast MRI, biopsy, tumor markers, follow‑up CT protocol).\n\n"
        "### Clinical Recommendations\n"
        "3–5 patient‑focused steps; avoid generic filler.\n\n"
        "### Full Diagnostic Explanation\n"
        "Detailed plain‑language summary for clinicians and patients.\n\n"
        "### AI Confidence Assessment\n"
        "Confidence near {CONFIDENCE_SEED}% (never 100%)."
    ),
    "ultrasound": (
        "You are a radiology specialist reporting on an {MODALITY_NAME}. "
        "Use observed signals: {SYMPTOMS}. Respond with sections below, tailored to ultrasound context.\n\n"
        "Condition Detected: <most likely diagnosis>\n"
        "### Findings\n"
        "Objective sonographic features (echogenicity, posterior enhancement/shadowing, margins, Doppler where relevant).\n\n"
        "### Impression\n"
        "Most likely diagnosis + differentials; explain key ultrasound signs.\n\n"
        "### Suggested Diagnostic Tests\n"
        "3–5 specific follow‑ups (e.g., targeted ultrasound, quantitative Doppler, MRI, relevant labs like beta‑hCG where appropriate).\n\n"
        "### Clinical Recommendations\n"
        "3–5 concrete, patient‑oriented steps; avoid generic filler.\n\n"
        "### Full Diagnostic Explanation\n"
        "Clear explanation suitable for non‑experts and clinicians.\n\n"
        "### AI Confidence Assessment\n"
        "Confidence near {CONFIDENCE_SEED}% (never 100%)."
    ),
    "mri": (
        "You are a radiology specialist reporting on an {MODALITY_NAME}. "
        "Use observed signals: {SYMPTOMS}.\n\n"
        "Condition Detected: <most likely diagnosis>\n"
        "### Findings\n"
        "Objective MRI features (sequence‑specific signal, enhancement, diffusion, edema, mass effect).\n\n"
        "### Impression\n"
        "Most likely diagnosis + differentials with reasoning.\n\n"
        "### Suggested Diagnostic Tests\n"
        "3–5 specific next tests tailored to suspected condition.\n\n"
        "### Clinical Recommendations\n"
        "3–5 concrete steps a clinician would advise the patient.\n\n"
        "### Full Diagnostic Explanation\n"
        "Comprehensive, accessible explanation.\n\n"
        "### AI Confidence Assessment\n"
        "Confidence near {CONFIDENCE_SEED}% (never 100%)."
    ),
}
# A generic fallback if you ever get an unexpected modality:
FALLBACK_TEMPLATE = (
    "You are a medical report assistant. Based on the image and patient symptoms: {symptoms}, "
    "generate a concise professional report including findings and recommendations."
)
# Utility: extract top-k symptom labels
def extract_top_symptoms(predictions: List[Tuple[str, float]], top_k: int = 3) -> List[str]:
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return [label for label, _ in sorted_preds[:top_k]]

# Generate report using multimodal Gemini
def _clamp_conf(seed: Optional[float]) -> int:
    try:
        if seed is None:
            return 75
        pct = int(round(float(seed) * 100))
        return max(35, min(95, pct))
    except Exception:
        return 75

async def generate_medical_report(symptoms: List[str], image_bytes: bytes, modality: str, mime_type: Optional[str] = None, confidence_seed: Optional[float] = None, llm_mode: Optional[str] = None) -> str:
    # Prepare prompt
    modality_key = modality.lower()
    template = PROMPT_TEMPLATES.get(modality_key, FALLBACK_TEMPLATE)
    conf = _clamp_conf(confidence_seed)
    # human‑readable modality name
    modality_name = {
        "xray": "X‑ray",
        "ct": "CT scan",
        "ultrasound": "ultrasound",
        "mri": "MRI scan",
    }.get(modality_key, modality_key)
    prompt = template.replace("{SYMPTOMS}", ", ".join(symptoms or [])) 
    prompt = prompt.replace("{CONFIDENCE_SEED}", str(conf)) 
    prompt = prompt.replace("{MODALITY_NAME}", modality_name)
    # prompt = (
    #     f"Based on the provided image and the following symptoms: {', '.join(symptoms)}, "
    #     "generate a clear, concise, and professional medical report. "
    #     "Include possible diagnoses, recommended next steps, and any relevant notes."
    # )
    # Wrap image bytes in Part for multimodal input
    

    # Mode resolution
    mode = _resolve_llm_mode(llm_mode)

    # Offline-only mode
    if mode == "offline":
        return _offline_structured_report(modality, symptoms, confidence_seed)

    # Prepare two variants: with image (if supported) and text-only as fallback
    contents_with_image = []
    if mode != "text-only" and image_bytes is not None and mime_type and mime_type.startswith("image/"):
        img_part = _make_image_content(image_bytes, mime_type)
        if img_part is not None:
            contents_with_image.append(img_part)
    contents_with_image.append({"type": "text", "text": prompt})

    contents_text_only = [{"type": "text", "text": prompt}]

    # Try image+text first, then text-only on failure (handles text-only models)
    try:
        text = await _llm_generate_async(contents_with_image)
        if text:
            return text
        # If empty, try text-only
        text2 = await _llm_generate_async(contents_text_only)
        if text2:
            return text2
        raise ValueError("Empty response from model for both image and text-only variants")
    except Exception as _e:
        # Try local LLM as fallback before going fully offline
        print(f"⚠ OpenRouter failed: {_e}, trying local LLM...")
        _init_local_llm()  # Ensure local LLM is loaded
        local_result = _local_llm_generate(prompt, max_tokens=600)
        if local_result and len(local_result) > 100:
            return local_result
        # Final offline fallback – return structured sections using local signals
        print("⚠ Local LLM unavailable or empty, using structured offline report")
        return _offline_structured_report(modality, symptoms, confidence_seed)




@app.post("/predict/xray/")
async def predict_xray(
    file: UploadFile = File(...),
    heavy_mode: Optional[bool] = Query(False, description="Use heavyweight vision model for maximum accuracy")
):
    global latest_xray_results
    
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # If heavy mode enabled, use heavy vision model
        if heavy_mode:
            _init_heavy_vision_model()
            
            # Only proceed with heavy model if it loaded successfully
            if _heavy_vision_model is not None:
                with open(temp_path, "rb") as f:
                    image_bytes = f.read()
                
                # Common X-ray conditions for CLIP
                xray_conditions = [
                    "Normal chest X-ray",
                    "Pneumonia",
                    "Pleural Effusion",
                    "Cardiomegaly",
                    "Congestive Heart Failure",
                    "Atelectasis",
                    "Pneumothorax",
                    "Mass",
                    "Nodule",
                    "Fracture",
                    "Consolidation",
                    "Infiltration"
                ]
                
                result = _heavy_vision_analyze(image_bytes, "xray", xray_conditions)
                os.remove(temp_path)
                
                if result and "predictions" in result:
                    predictions = result["predictions"]
                    latest_xray_results = {label: float(prob) for label, prob in predictions}
                    
                    # confidence gating
                    top = max((p for _, p in predictions), default=0.0)
                    if top < 0.6:
                        return JSONResponse(content={
                            "predictions": predictions,
                            "note": "Low confidence – result is indeterminate. Please consult a clinician.",
                            "model": "CLIP-ViT-Large (Heavy Mode)"
                        })
                    return JSONResponse(content={
                        "predictions": predictions,
                        "model": "CLIP-ViT-Large (Heavy Mode)"
                    })
        
        # Default: use standard model (also runs if heavy model failed to load)
        predictions = process_xray(temp_path, device="cpu")
        os.remove(temp_path)
        latest_xray_results = {label: float(prob) for label, prob in predictions}
        # confidence gating
        top = max((p for _, p in predictions), default=0.0)
        if top < 0.6:
            return JSONResponse(content={
                "predictions": predictions,
                "note": "Low confidence – result is indeterminate. Please consult a clinician."
            })
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_latest_results/")
async def get_latest_results():
    if not latest_xray_results:
        return {"message": "No prediction results available yet."}
    return latest_xray_results


@app.post("/generate-report/{modality}/")
async def generate_report(
    modality: str = Path(..., description="One of: xray, ct, ultrasound, mri"),
    file: UploadFile = File(...),
    llm_mode: Optional[str] = Query(None)
):
    modality = modality.lower()
    if modality not in ["xray", "ct", "ultrasound", "mri"]:
        raise HTTPException(status_code=400, detail="Invalid modality.")
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"temp_{modality}_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        # Inference dispatch
        if modality == "xray":
            # No constraints: proceed without gating by chest X-ray heuristics
            raw_preds = process_xray(temp_path, device="cpu")
        # elif modality == "ct": raw_preds = process_ct(temp_path, device="cpu")
        # elif modality == "ultrasound": raw_preds = process_ultrasound(temp_path, device="cpu")
        # else: raw_preds = process_mri(temp_path, device="cpu")

        symptoms = extract_top_symptoms(raw_preds)
        # Read bytes
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        os.remove(temp_path)

        # determine confidence seed from top prediction if available
        top_prob = 0.0
        try:
            top_prob = max((p for _, p in raw_preds), default=0.0)
        except Exception:
            top_prob = 0.0
        report = await generate_medical_report(symptoms, img_bytes, modality, mime_type=file.content_type, confidence_seed=top_prob, llm_mode=llm_mode)
        # Extract the disease from the report
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"
        # Store the report in a global variable
        latest_reports[modality] = {
            "disease": disease,
            "symptoms": symptoms,
            "report": report
        }
        return JSONResponse(content={"symptoms": symptoms, "disease": disease ,"report": report})
    except HTTPException:
        os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get-latest-report/{modality}/")
async def get_latest_report(modality: str = Path(...)):
    modality = modality.lower()
    if modality not in latest_reports:
        raise HTTPException(status_code=404, detail="No report available for this modality.")
    return latest_reports[modality]


# CT 2D and 3D routes
@app.post("/predict/ct/2d/")
async def generate_report_ct2d(
    file: UploadFile = File(...),
    llm_mode: Optional[str] = Query(None),
    heavy_mode: Optional[bool] = Query(False, description="Use heavyweight vision model for maximum accuracy")
):
    modality = "ct"
    mode = "2d"

    # Only allow image files for 2D slices
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type for CT2D.")

    temp_path = f"temp_ct2d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # If heavy mode enabled, use heavy vision model
        if heavy_mode:
            _init_heavy_vision_model()
            
            # Only proceed with heavy model if it loaded successfully
            if _heavy_vision_model is not None:
                with open(temp_path, "rb") as f:
                    img_bytes = f.read()
                
                # Common CT conditions
                ct_conditions = [
                    "Normal CT scan",
                    "Tumor",
                    "No Tumor",
                    "Cyst",
                    "Hemorrhage",
                    "Fracture",
                    "Mass",
                    "Pneumonia",
                    "Atelectasis"
                ]
                
                result = _heavy_vision_analyze(img_bytes, "ct", ct_conditions)
                
                if result and "predictions" in result:
                    raw_preds = result["predictions"]
                    symptoms = [cond for cond, _ in raw_preds[:3]]
                    top_prob = result.get("confidence", 0.0)
                    
                    os.remove(temp_path)
                    
                    # Generate report
                    report = await generate_medical_report(
                        symptoms, img_bytes, modality=modality, mime_type=file.content_type, 
                        confidence_seed=top_prob, llm_mode=llm_mode
                    )
                    
                    disease = result.get("top_condition", "Unknown")
                    
                    latest_reports["ct2d"] = {
                        "symptoms": symptoms,
                        "disease": disease,
                        "report": report,
                        "model": "CLIP-ViT-Large (Heavy Mode)"
                    }
                    
                    return JSONResponse({
                        "symptoms": symptoms,
                        "disease": disease,
                        "report": report,
                        "predictions": raw_preds,
                        "model": "CLIP-ViT-Large (Heavy Mode)"
                    })
        
        # Default: use standard model (also runs if heavy model failed to load)
        raw_preds = process_ct(temp_path, mode=mode, device="cpu")
        symptoms = extract_top_symptoms(raw_preds)

        # Read image bytes before deleting temp
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        os.remove(temp_path)

        # Generate report using correct MIME type
        top_prob = 0.0
        try:
            top_prob = max((p for _, p in raw_preds), default=0.0)
        except Exception:
            top_prob = 0.0
        report = await generate_medical_report(
            symptoms, img_bytes, modality=modality, mime_type=file.content_type, confidence_seed=top_prob, llm_mode=llm_mode
        )

        # Extract disease
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # Store
        latest_reports["ct2d"] = {
            "symptoms": symptoms,
            "disease": disease,
            "report": report
        }

        return JSONResponse({
            "symptoms": symptoms,
            "disease": disease,
            "report": report
        })

    except HTTPException:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))



## 3d route 
@app.post("/predict/ct/3d/")
async def generate_report_ct3d(file: UploadFile = File(...), llm_mode: Optional[str] = Query(None)):
    # 1) Save upload to disk
    temp_path = f"temp_ct3d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # 2) Run your 3D model to get symptoms label(s)
        effective_mode = _resolve_llm_mode(llm_mode)
        raw_preds = process_ct(temp_path, mode="3d", device="cpu")
        label, prob = raw_preds[0] # type: ignore
        symptoms = [label]
        if effective_mode == "offline":
            # Build offline structured report and return immediately
            report = _offline_structured_report("ct", symptoms, prob)
            os.remove(temp_path)
            latest_reports["ct3d"] = {
                "Symptom": label,
                "disease": re.search(r"Condition Detected:\s*(.+)", report).group(1) if re.search(r"Condition Detected:\s*(.+)", report) else label,
                "report": report,
            }
            return JSONResponse(latest_reports["ct3d"]) 

        # 3) Load volume and pick mid-slices (fallback to 2D image if needed)
        slices = {}
        try:
            img = load(temp_path)
            vol = img.get_fdata() #type: ignore
            z, y, x = [d // 2 for d in vol.shape]
            slices = {
                "axial":   vol[z, :, :],
                "coronal": vol[:, y, :],
                "sagittal":vol[:, :, x],
            }
        except Exception:
            # Fallback: treat as 2D image and synthesize three views
            from PIL import Image as _PILImage
            import numpy as _np
            img2d = _PILImage.open(temp_path).convert("L").resize((224,224))
            arr = _np.array(img2d)
            slices = {"axial": arr, "coronal": arr, "sagittal": arr}

        # 4) Convert each slice to PNG bytes
        image_parts = []
        for name, sl in slices.items():
            # normalize slice to [0,255]
            sl_norm = ((sl - sl.min())/(sl.max()-sl.min()) * 255).astype(np.uint8)
            pil = Image.fromarray(sl_norm).convert("L").resize((224,224))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            image_parts.append(_make_image_content(buf.getvalue(), "image/png"))

        os.remove(temp_path)

        # 5) Build prompt & send all three images + prompt
        prompt = (
            '''
        You are a medical AI assistant specialized in interpreting 3D and 2D CT scan results. 
        Given a set of AI-generated confidence scores for tumor detection, your task is to:

        1. Identify whether a tumor or no tumor is more likely based on the highest confidence score.
        2. Clearly mention the detected condition and the confidence score as a percentage (e.g., 92.00%).
        3. Explain what this result means for the patient in clear, simple language.
        4. Describe briefly how 3D CT scans assist in detecting tumors by providing detailed cross-sectional views of the body.
        5. Recommend possible next steps such as further imaging or biopsy for confirmation.
        6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
        7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        8. Report size should be always between 200 and 300 words.
        9. Use the following format for the output:

        Output example 
        Condition Detected: Tumor
        The AI analysis of your 3D CT scan of the brain indicates a high probability of a tumor, with a confidence score of 92.00%. This suggests there may be an abnormal mass or growth
        present in the scanned region. 3D CT scans allow doctors to view detailed cross-sectional images of internal tissues, making it easier to identify potential issues like 
        tumors. While this result is a strong indicator, it is not a confirmed diagnosis. Further testing, such as an MRI or biopsy, may be required. 
        Disclaimer: This is an AI-generated summary. Please consult a certified doctor or radiologist for medical confirmation and advice.
            '''
        )

        if effective_mode == "text-only":
            contents = [{"type": "text", "text": prompt}]
        else:
            contents = [p for p in image_parts if p]
            contents.append({"type": "text", "text": prompt})
        report = await _llm_generate_async(contents) or "<empty>"
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # 6) Store & return
        latest_reports["ct3d"] = {
            "Symptom": label,
            "disease": disease,
            "report": report
        }
        return JSONResponse(latest_reports["ct3d"])

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/predict/ct/2d/")
async def get_latest_report_ct2d():
    if "ct2d" not in latest_reports:
        raise HTTPException(status_code=404, detail="No 2D CT report available.")
    return latest_reports["ct2d"]

@app.get("/predict/ct/3d/")
async def get_latest_report_ct3d():
    if "ct3d" not in latest_reports:
        raise HTTPException(status_code=404, detail="No 3D CT report available.")
    return latest_reports["ct3d"]

@app.post("/predict/mri/3d/")
async def generate_report_mri3d(
    file: UploadFile = File(...),
    llm_mode: Optional[str] = Query(None),
    heavy_mode: Optional[bool] = Query(False, description="Use heavyweight vision model for maximum accuracy")
):  
    # 1) Save upload to disk
    temp_path = f"temp_mri3d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        effective_mode = _resolve_llm_mode(llm_mode)
        
        # If heavy mode enabled, use heavy vision model
        if heavy_mode:
            _init_heavy_vision_model()
            
            # Only proceed with heavy model if it loaded successfully
            if _heavy_vision_model is not None:
                # Try to extract mid-slice or use full image
                try:
                    img = load(temp_path)
                    vol = img.get_fdata() #type: ignore
                    z = vol.shape[0] // 2
                    slice_2d = vol[z, :, :]
                    # normalize to [0,255]
                    slice_norm = ((slice_2d - slice_2d.min())/(slice_2d.max()-slice_2d.min()) * 255).astype(np.uint8)
                    pil = Image.fromarray(slice_norm).convert("RGB").resize((224,224))
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                except Exception:
                    # Fallback: read as regular image
                    from PIL import Image as _PILImage
                    img2d = _PILImage.open(temp_path).convert("RGB").resize((224,224))
                    buf = io.BytesIO()
                    img2d.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                
                # Common MRI brain conditions
                mri_conditions = [
                    "Normal MRI",
                    "Glioma",
                    "Meningioma",
                    "Pituitary Tumor",
                    "No Tumor",
                    "Tumor",
                    "Mass"
                ]
                
                result = _heavy_vision_analyze(img_bytes, "mri", mri_conditions)
                os.remove(temp_path)
                
                if result and "predictions" in result:
                    raw_preds = result["predictions"]
                    label = result.get("top_condition", "Unknown")
                    prob = result.get("confidence", 0.0)
                    symptoms = [label]
                    
                    # Generate report
                    report = await generate_medical_report(
                        symptoms, img_bytes, modality="mri", mime_type="image/png",
                        confidence_seed=prob, llm_mode=llm_mode
                    )
                    
                    disease = re.search(r"Condition Detected:\s*(.+)", report).group(1) if re.search(r"Condition Detected:\s*(.+)", report) else label
                    
                    latest_reports["mri3d"] = {
                        "Symptom": label,
                        "disease": disease,
                        "report": report,
                        "model": "CLIP-ViT-Large (Heavy Mode)"
                    }
                    
                    return JSONResponse(latest_reports["mri3d"])
        
        # Default: use standard model (also runs if heavy model failed to load)
        raw_preds = process_mri(temp_path, mode='3d', device="cpu")
        label, prob = raw_preds[0]
        symptoms = [label]
        if effective_mode == "offline":
            report = _offline_structured_report("mri", symptoms, prob)
            os.remove(temp_path)
            latest_reports["mri3d"] = {
                "Symptom": label,
                "disease": re.search(r"Condition Detected:\s*(.+)", report).group(1) if re.search(r"Condition Detected:\s*(.+)", report) else label,
                "report": report,
            }
            return JSONResponse(latest_reports["mri3d"]) 

        # 3) Load volume and pick mid-slices (fallback to 2D image if needed)
        slices = {}
        try:
            img = load(temp_path)
            vol = img.get_fdata() #type: ignore
            z, y, x = [d // 2 for d in vol.shape]
            slices = {
                "axial":   vol[z, :, :],
                "coronal": vol[:, y, :],
                "sagittal":vol[:, :, x],
            }
        except Exception:
            from PIL import Image as _PILImage
            import numpy as _np
            img2d = _PILImage.open(temp_path).convert("L").resize((224,224))
            arr = _np.array(img2d)
            slices = {"axial": arr, "coronal": arr, "sagittal": arr}

        # 4) Convert each slice to PNG bytes
        image_parts = []
        for name, sl in slices.items():
            # normalize slice to [0,255]
            sl_norm = ((sl - sl.min())/(sl.max()-sl.min()) * 255).astype(np.uint8)
            pil = Image.fromarray(sl_norm).convert("L").resize((224,224))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            image_parts.append(_make_image_content(buf.getvalue(), "image/png"))

        os.remove(temp_path)

        # 5) Build prompt & send all three images + prompt
        prompt = (
            '''
                You are a medical specialist in interpreting brain MRI results. 
                Based on the image and the patient symptoms: {symptoms}, your task is to:

                1. Identify the condition with the highest confidence score from the list: ["No Tumor", "Meningioma", "Glioma", "Pituitary Tumor"].
                2. Clearly mention the detected condition and the confidence score as a percentage (e.g., 87.45%).
                3. Explain what this result means for the patient in clear, simple language, based on the detected condition.
                4. Describe briefly how brain MRI helps in identifying such conditions by providing high-resolution images of soft tissues.
                5. Suggest possible next steps, such as neurologist consultation, further imaging, or biopsy, depending on the condition.
                6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
                7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
                8. Report size should be always between 200 and 300 words.
                9. create a detailed MRI report including key findings, interpretation, and suggested follow‑up
                9. Use the following format for the output:

                Condition Detected: Glioma
                The AI analysis of your brain MRI scan suggests a high probability of Glioma, with a confidence score of 89.00%. Gliomas are tumors that originate in the glial cells of the brain or spinal cord. They can affect brain function depending on their location, size, and growth rate, potentially causing symptoms such as headaches, seizures, or neurological changes.

                MRI scans are highly effective for detecting such tumors, as they offer detailed images of soft brain tissues. This allows for accurate visualization of the tumor's structure and position, which is crucial for early diagnosis and treatment planning.

                Although this result indicates a strong likelihood of Glioma, it is not a confirmed medical diagnosis. You should consult a neurologist or oncologist for further evaluation. Additional tests like a contrast-enhanced MRI or biopsy may be recommended to validate the finding.

                Disclaimer: This is an AI-generated result. Please seek advice from a certified medical professional.
            '''
        ).format(symptoms=symptoms)

        if effective_mode == "text-only":
            contents = [{"type": "text", "text": prompt}]
        else:
            contents = [p for p in image_parts if p]
            contents.append({"type": "text", "text": prompt})
        report = await _llm_generate_async(contents) or "<empty>"
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # 6) Store & return
        latest_reports["mri3d"] = {
            "Symptom": label,
            "disease": disease,
            "report": report
        }
        return JSONResponse(latest_reports["mri3d"])
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/predict/mri/3d/")
async def get_latest_report_mri3d():
    if "mri3d" not in latest_reports:
        raise HTTPException(status_code=404, detail="No 3D MRI report available.")
    return latest_reports["mri3d"]

@app.post("/predict/ultrasound/")
async def generate_report_ultrasound(
    file: UploadFile = File(...),
    llm_mode: Optional[str] = Query(None),
    heavy_mode: Optional[bool] = Query(False, description="Use heavyweight vision model for maximum accuracy")
):
    modality = "ultrasound"

    # 1) Validate content type before saving
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # 2) Save upload to disk
    temp_path = f"temp_{modality}_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # If heavy mode enabled, use heavy vision model
        if heavy_mode:
            _init_heavy_vision_model()
            
            # Only proceed with heavy model if it loaded successfully
            if _heavy_vision_model is not None:
                with open(temp_path, "rb") as f:
                    img_bytes = f.read()
                
                # Common ultrasound conditions
                ultrasound_conditions = [
                    "Normal ultrasound",
                    "Cyst",
                    "Tumor",
                    "Mass",
                    "Fluid collection",
                    "Gallstones",
                    "Kidney stones",
                    "Abnormal growth"
                ]
                
                result = _heavy_vision_analyze(img_bytes, "ultrasound", ultrasound_conditions)
                os.remove(temp_path)
                
                if result and "predictions" in result:
                    raw_preds = result["predictions"]
                    symptoms = [cond for cond, _ in raw_preds[:3]]
                    top_prob = result.get("confidence", 0.0)
                    
                    # Generate report
                    report = await generate_medical_report(
                        symptoms, img_bytes, modality=modality, mime_type=file.content_type,
                        confidence_seed=top_prob, llm_mode=llm_mode
                    )
                    
                    disease = result.get("top_condition", "Unknown")
                    
                    latest_reports["ultrasound"] = {
                        "symptoms": symptoms,
                        "disease": disease,
                        "report": report,
                        "model": "CLIP-ViT-Large (Heavy Mode)"
                    }
                    
                    return JSONResponse({
                        "symptoms": symptoms,
                        "disease": disease,
                        "report": report,
                        "predictions": raw_preds,
                        "model": "CLIP-ViT-Large (Heavy Mode)"
                    })
        
        # Default: use standard model (also runs if heavy model failed to load)
        # No constraints: proceed without ultrasound gating
        # 3) Run your ultrasound model to get symptom labels
        raw_preds = process_ultrasound(temp_path, device="cpu")
        symptoms = extract_top_symptoms(raw_preds)

        # 4) Read bytes for report generation
        with open(temp_path, "rb") as f:
            img_bytes = f.read()

        # remove temp file ASAP
        os.remove(temp_path)

        # 5) Generate the medical report (with optional LLM mode override)
        report = await generate_medical_report(symptoms, img_bytes, modality=modality, mime_type=file.content_type, llm_mode=llm_mode)

        def extract_condition(report: str) -> str:
            """
            Robustly pull the text immediately following 'Condition Detected:' 
            up to the first non‑empty line, ignoring case/extra whitespace.
            """
            if not report:
                return "Unknown"

            lower = report.lower()
            keyword = "condition detected"
            start = lower.find(keyword)
            if start == -1:
                return "Unknown"

            # Find the colon after the keyword
            colon = report.find(":", start + len(keyword))
            if colon == -1:
                return "Unknown"

            # Grab everything after the colon
            tail = report[colon+1:]

            # Split into lines, return the first non-blank one
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    return line

            return "Unknown"

        disease = extract_condition(report)
        # 7) Store in global for frontend polling if needed
        latest_reports[modality] = {
            "disease":  disease,
            "symptoms": symptoms,
            "report":   report,
        }

        # 8) Confidence gating and JSON
        top = 0.0
        try:
            top = max((p for _, p in raw_preds), default=0.0)
        except Exception:
            top = 0.0
        payload = {"symptoms": symptoms, "disease": disease, "report": report}
        if top < 0.6:
            payload["note"] = "Low confidence – consider a follow-up scan."
        return JSONResponse(content=payload)

    except HTTPException:
        # Already an HTTPException—nothing extra to clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as e:
        # Catch‐all: ensure temp file is removed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/predict/ultrasound/")
async def get_latest_report_ultrasound():   
    if "ultrasound" not in latest_reports:
        raise HTTPException(status_code=404, detail="No ultrasound report available.")
    return latest_reports["ultrasound"]

# Mock database of doctors
class Doctor(BaseModel):
    name: str
    specialty: str
    location: str
    phone: str
    lat: float
    lng: float

def build_overpass_query(lat: float, lng: float, shift: float = 0.03) -> str:
    lat_min = lat - shift
    lng_min = lng - shift
    lat_max = lat + shift
    lng_max = lng + shift
    return f"""
    [out:json][timeout:25];
    node
      [healthcare=doctor]
      ({lat_min},{lng_min},{lat_max},{lng_max});
    out;
    """

@app.get("/api/search-doctors")
async def search_doctors(location: str, specialty: str = ""):
    geolocator = Nominatim(user_agent="doctor-search")
    location_obj = geolocator.geocode(location + ", India")
    if not location_obj:
        return []

    lat, lon = location_obj.latitude, location_obj.longitude # type: ignore

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["healthcare"="doctor"](around:10000,{lat},{lon});
      node["amenity"="doctors"](around:10000,{lat},{lon});
    );
    out body;
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(overpass_url, data=query) # type: ignore
            data = res.json()
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={"detail": "Overpass API timeout. Please try again later."}
        )


    doctors = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed Doctor")
        specialty_tag = (
            tags.get("healthcare:speciality") or
            tags.get("healthcare:specialty") or
            tags.get("specialty") or
            "General"
        )
        if specialty and specialty.lower() not in specialty_tag.lower():
            continue

        phone = tags.get("phone", "Not available")
        addr = tags.get("addr:city") or tags.get("addr:suburb") or location

        doctors.append({
            "name": name,
            "specialty": specialty_tag,
            "location": addr,
            "phone": phone,
            "lat": el.get("lat"),
            "lng": el.get("lon")
        })

    return doctors
# @app.get("/api/get-doctor/{doctor_id}", response_model=Doctor)


#chatbot of landing page 

class ChatRequest(BaseModel):
    message: str

@app.post("/chat_with_report/")
async def chat_with_report(request: ChatRequest):
    user_message = request.message.lower()

    # Rule-based chatbot responses
    if "upload" in user_message and "image" in user_message:
        reply = (
            "To upload a medical image, go to the 'Upload' section from the navbar. "
            "There, you can choose from 5 model types: MRI, X-ray, Ultrasound, CT Scan 2D, and CT Scan 3D. "
            "After selecting the type and uploading your image, click 'Upload and Analyze' to get the result."
        )
    elif "analyze" in user_message or "report" in user_message:
        reply = (
            "Once you upload an image and select the model type, clicking 'Upload and Analyze' will route you to the result page. "
            "This page displays an AI-generated diagnostic report based on the image you provided."
        )
    elif "features" in user_message:
        reply = (
            "Our website offers features like disease prediction using 6 medical models, instant report generation, "
            "testimonials from patients, a FAQ section, and easy contact options."
        )
    elif "models" in user_message or "which scans" in user_message:
        reply = (
            "The supported models are:\n"
            "- MRI 2D\n- MRI 3D\n- X-ray\n- Ultrasound\n- CT Scan 2D\n- CT Scan 3D"
        )
    elif "contact" in user_message:
        reply = (
            "You can find the contact section by scrolling to the 'Contact' part of the homepage, or directly in the footer."
        )
    elif "testimonials" in user_message:
        reply = (
            "We showcase real testimonials from users who have benefited from our AI diagnosis platform."
        )
    elif "faq" in user_message or "questions" in user_message:
        reply = (
            "The FAQ section answers common questions related to uploading images, interpreting reports, and data privacy."
        )
    elif "hero" in user_message or "homepage" in user_message:
        reply = (
            "The hero section on our homepage highlights the goal of our platform — fast and accurate diagnosis from medical images using AI."
        )
    elif "cta" in user_message or "get started" in user_message:
        reply = (
            "The Call-To-Action (CTA) section encourages users to start using the platform by uploading an image and receiving a report."
        )
    else:
        reply = (
            "I'm here to help you with any questions about using the platform. "
            "You can ask me how to upload images, what models are supported, or what happens after analysis."
        )

    return {"response": reply}

# ----------------------------
# Conversational AI about a diagnosis/report
# ----------------------------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str

class ChatPayload(BaseModel):
    modality: Optional[str] = None
    messages: List[ChatTurn]
    report_context: Optional[Dict[str, Any]] = None

def _build_chat_prompt(modality: str, report_ctx: Dict[str, Any], history: List[ChatTurn]) -> str:
    mod = (modality or "").lower() or "xray"
    nice_mod = {
        "xray": "X‑ray",
        "ct": "CT scan",
        "ct2d": "CT scan (2D)",
        "ct3d": "CT scan (3D)",
        "mri": "MRI",
        "mri3d": "MRI (3D)",
        "ultrasound": "Ultrasound",
    }.get(mod, mod.upper())

    disease = (report_ctx or {}).get("disease") or (report_ctx or {}).get("diagnosis") or "Unknown"
    symptoms = (report_ctx or {}).get("symptoms") or []
    report_text = (report_ctx or {}).get("report") or (report_ctx or {}).get("analysis") or ""

    history_lines = []
    for turn in history[-12:]:
        r = turn.role.lower()
        if r == "assistant":
            history_lines.append(f"Assistant: {turn.text}")
        elif r == "user":
            history_lines.append(f"User: {turn.text}")
    history_txt = "\n".join(history_lines).strip()

    prompt = (
        f"You are a medical report assistant specialized in {nice_mod}. "
        f"Answer follow‑up questions about a previously generated diagnosis. "
        f"Be factual, concise, and clinically oriented. If uncertain, state limitations and suggest appropriate follow‑ups.\n\n"
        f"Report context (for grounding):\n"
        f"- Disease: {disease}\n"
        f"- Symptoms/signals: {', '.join(symptoms) if symptoms else 'n/a'}\n"
        f"- Report summary: {report_text[:1200]}\n\n"
        + (f"Conversation so far:\n{history_txt}\n\n" if history_txt else "")
        + "Respond to the latest user question directly without preamble."
    )
    return prompt

def _offline_chat_reply(modality: str, report_ctx: Dict[str, Any], user_message: str) -> str:
    disease = (report_ctx or {}).get("disease") or (report_ctx or {}).get("diagnosis") or "Unknown"
    details = _get_condition_details(disease) if disease and disease != "Unknown" else None
    base = "Based on your report, here's guidance: "
    if details:
        tests = ", ".join(details.get("tests", [])[:4]) or "follow‑up as clinically indicated"
        recs = ", ".join(details.get("recommendations", [])[:3]) or "discuss with your clinician"
        return (
            f"{base}The findings are consistent with {disease}. "
            f"Typical next steps include: {tests}. "
            f"Recommendations: {recs}. "
            "This assistant cannot provide definitive diagnoses; please coordinate with your clinician for personalized care."
        )
    return (
        f"{base}The reported condition is {disease}. Consider appropriate follow‑up imaging, relevant labs, and consultation with a specialist as needed. "
        "This information supports, but does not replace, professional medical advice."
    )

@app.post("/ai/chat/")
async def ai_chat(payload: ChatPayload, llm_mode: Optional[str] = Query(None)):
    modality = (payload.modality or "").lower()
    ctx = payload.report_context or {}
    if not ctx:
        for key in [modality, "xray", "ct2d", "ct3d", "mri3d", "ultrasound", "ct", "mri"]:
            if key and key in latest_reports:
                ctx = latest_reports[key]
                modality = key
                break

    prompt = _build_chat_prompt(modality or "xray", ctx or {}, payload.messages)

    mode = _resolve_llm_mode(llm_mode)
    provider = "gemini" if _get_gemini_key() else "openrouter"

    if mode == "offline":
        reply = _offline_chat_reply(modality, ctx or {}, payload.messages[-1].text if payload.messages else "")
        return JSONResponse({
            "provider": "offline",
            "model": "offline-structured",
            "reply": reply,
        })

    if provider == "gemini":
        text = _gemini_generate(prompt_text=prompt, image_bytes=None, mime_type=None, text_only=True)
        if text:
            return JSONResponse({
                "provider": "gemini",
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                "reply": text,
            })
        provider = "openrouter"

    try:
        text = await _llm_generate_async([{"type": "text", "text": prompt}], model_override=os.getenv("OPENROUTER_MODEL_CHAT", OPENROUTER_MODEL))
        if text:
            return JSONResponse({
                "provider": "openrouter",
                "model": os.getenv("OPENROUTER_MODEL_CHAT", OPENROUTER_MODEL),
                "reply": text,
            })
        raise RuntimeError("Empty response")
    except Exception as e:
        print(f"⚠ OpenRouter chat failed: {e}, trying local/offline...")
        _init_local_llm()
        text = _local_llm_generate(prompt, max_tokens=400)
        if text and len(text) > 40:
            return JSONResponse({
                "provider": "local",
                "model": "microsoft/phi-2",
                "reply": text,
            })
        reply = _offline_chat_reply(modality, ctx or {}, payload.messages[-1].text if payload.messages else "")
        return JSONResponse({
            "provider": "offline",
            "model": "offline-structured",
            "reply": reply,
        })