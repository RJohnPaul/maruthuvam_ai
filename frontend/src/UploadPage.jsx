import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from "./components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './components/ui/card';
import { Progress } from './components/ui/progress';
import { Alert, AlertDescription } from './components/ui/alert';
import { Upload, X, Check, AlertCircle, Image } from 'lucide-react';
import ImageTypeSelector from './components/ImageTypeSelector';
import SegmentedImageViewer from './components/SegmentedImageViewer';

const UploadPage = ({ selectedImageType, setSelectedImageType, setProcessedData, mockEnabled = false, aiEnabled = false, heavyModelEnabled = false }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setError(null);
    setFile(selectedFile);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const BASE_API_URL = 'http://127.0.0.1:8000';

  const pickMockCase = (name, type) => {
    const n = (name || '').toLowerCase();
    // Heuristics by filename and selected type
    if (n.includes('forearm') || n.includes('radius')) return 'forearm_xray';
    if (n.includes('ultra') || n.includes('obst') || n.includes('gestation')) return 'ob_ultrasound';
    if (n.includes('fetal') || n.includes('thumb') || n.includes('baby')) return 'fetal_ultrasound';
    if (n.includes('mra') || n.includes('angiography') || n.includes('willis')) return 'brain_mra_ok';
    if (n.includes('lesion') || n.includes('flair') || n.includes('t2')) return 'brain_mri_lesion';
    if (n.includes('chest') || n.includes('lungs') || n.includes('heart')) return 'chest_xray';
    if (n.includes('femur') || n.includes('tibia') || n.includes('post') || n.includes('rod')) return 'leg_xray_postop';
    // fallback by type
    if (type === 'xray') return 'chest_xray';
    if (type === 'ultrasound') return 'ob_ultrasound';
    if (type && type.startsWith('mri')) return 'brain_mri_lesion';
    return 'forearm_xray';
  };

  const buildMock = (key) => {
    switch (key) {
      case 'forearm_xray':
        return {
          predictions: [
            ['Radius Fracture (mid-shaft, non-displaced)', 0.97],
            ['Ulna Intact', 0.92],
            ['Normal Bone Density', 0.88],
          ],
          disease: 'Forearm X-ray — Mid-shaft radius fracture (non-displaced)',
          report:
            '1️⃣ Forearm X-ray (Fracture)\n\n' +
            'Findings: Mid-shaft fracture of the radius, clean and non-displaced.\n\n' +
            'Recommendations:\n' +
            '- Immobilize with a forearm cast or splint immediately.\n' +
            '- Avoid any weight-bearing or lifting till ortho clearance.\n' +
            '- Manage pain with NSAIDs (e.g., ibuprofen, if not contraindicated).\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Repeat X-ray after 1–2 weeks → check bone alignment and healing.\n' +
            '- Neurovascular exam of the limb (check for sensation, pulse, and motor).\n' +
            '- If pain/swelling worsens → MRI to rule out soft tissue or ligament injury.',
          symptoms: ['forearm pain', 'swelling', 'tenderness'],
        };
      case 'ob_ultrasound':
        return {
          predictions: null,
          disease: 'Ultrasound — Early gestational sac (~5 weeks)',
          report:
            '2️⃣ Ultrasound (Early Gestational Sac)\n\n' +
            'Findings: Small gestational sac; early intrauterine pregnancy (~5 weeks).\n\n' +
            'Recommendations:\n' +
            '- Confirm normal pregnancy development.\n' +
            '- Begin prenatal vitamins + schedule regular antenatal visits.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Repeat ultrasound in 1–2 weeks → confirm fetal pole + heartbeat.\n' +
            '- Serum β-hCG levels → should double roughly every 48–72 hours.\n' +
            '- Progesterone levels → check for viable pregnancy support.\n' +
            '- Blood group + Rh typing (standard early pregnancy protocol).',
          symptoms: ['missed period', 'nausea', 'mild cramping'],
        };
      case 'brain_mra_ok':
        return {
          predictions: null,
          disease: 'Brain MRI/MRA — Normal vessels',
          report:
            '3️⃣ Brain MRI/MRA (Normal Vessels)\n\n' +
            'Findings: Circle of Willis looks intact; no aneurysm or blockage.\n\n' +
            'Recommendations:\n' +
            '- Maintain regular BP and cholesterol levels to protect those vessels.\n' +
            '- If done for headaches or dizziness → monitor symptoms, rule out other causes.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Neurological exam → confirm no deficits.\n' +
            '- MRV (Magnetic Resonance Venography) if venous issue suspected.\n' +
            '- EEG if seizures are part of symptoms.\n' +
            '- CT Angiography (CTA) if MRA findings are unclear.',
          symptoms: ['headache (if present)', 'dizziness (if present)'],
        };
      case 'brain_mri_lesion':
        return {
          predictions: null,
          disease: 'Brain MRI — Single bright lesion on FLAIR',
          report:
            '4️⃣ Brain MRI (Lesion/Abnormal Bright Spot)\n\n' +
            'Findings: Single bright lesion on FLAIR — could be cyst, infection, tumor, or demyelination.\n\n' +
            'Recommendations:\n' +
            '- Immediate referral to a neurologist.\n' +
            '- Don’t ignore even if symptoms mild — early eval = better outcomes.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Contrast MRI brain → to check enhancement pattern.\n' +
            '- MR Spectroscopy / Diffusion MRI → to differentiate tumor vs abscess.\n' +
            '- Blood tests: CBC, ESR, CRP → for infection or inflammation.\n' +
            '- Lumbar puncture (CSF analysis) if infection/demyelination suspected.\n' +
            '- Neurosurgical consult if lesion size/growth confirmed.',
          symptoms: ['headache', 'neurologic symptoms (if present)'],
        };
      case 'fetal_ultrasound':
        return {
          predictions: null,
          disease: 'Ultrasound – Fetal (Thumb-Sucking Baby)',
          report:
            '5️⃣ Ultrasound – Fetal (Thumb-Sucking Baby)\n\n' +
            'Findings:\n' +
            '- Standard second-trimester fetal ultrasound.\n' +
            '- Clear visualization of fetal head, spine, and limbs.\n' +
            '- Normal amniotic fluid index.\n' +
            '- Fetal activity present (thumb-sucking = normal neurobehavioral sign).\n\n' +
            'Interpretation:\n' +
            'Normal fetal development; no gross congenital anomalies visible in this single still image.\n\n' +
            'Recommendations:\n' +
            '- Continue routine antenatal checkups.\n' +
            '- Maintain proper maternal nutrition and hydration.\n' +
            '- Continue folic acid and iron supplements.\n' +
            '- Avoid smoking, alcohol, and stress.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Anomaly scan (18–22 weeks) – check organ systems.\n' +
            '- Fetal growth scan every 4–6 weeks.\n' +
            '- Maternal serum screening for genetic abnormalities.\n' +
            '- Glucose tolerance test (GTT) – screen for gestational diabetes.',
          symptoms: ['pregnancy follow-up'],
        };
      case 'chest_xray':
        return {
          predictions: [
            ['Right lower lobe pneumonia vs mild effusion', 0.78],
            ['No rib fracture', 0.9],
            ['Normal cardiac silhouette', 0.86],
          ],
          disease: 'Chest X-ray – Lungs and Heart',
          report:
            '6️⃣ Chest X-ray – Lungs and Heart\n\n' +
            'Findings:\n' +
            '- Patchy opacity in the right lower lung field, possible pneumonia or mild pleural effusion.\n' +
            '- No visible rib fractures.\n' +
            '- Cardiac silhouette normal; no cardiomegaly or major deformity.\n\n' +
            'Interpretation:\n' +
            'Suggestive of infective consolidation (pneumonia) or fluid accumulation at the base of the right lung.\n\n' +
            'Recommendations:\n' +
            '- Antibiotic therapy if pneumonia suspected.\n' +
            '- Maintain hydration and rest.\n' +
            '- Avoid exposure to pollutants/smoke.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- CBC + ESR/CRP → infection markers.\n' +
            '- Sputum culture and sensitivity → confirm pathogen.\n' +
            '- Repeat chest X-ray after 10–14 days.\n' +
            '- Chest CT scan → if opacity persists or TB suspected.',
          symptoms: ['cough', 'fever', 'breathlessness'],
        };
      case 'leg_xray_postop':
        return {
          predictions: [
            ['Healing post-operative femoral fracture', 0.9],
            ['Stable fixation hardware', 0.88],
          ],
          disease: 'Leg X-ray (Femur/Tibia) — Post-surgery',
          report:
            '7️⃣ Leg X-ray (Femur/Tibia – Post-Surgery)\n\n' +
            'Findings:\n' +
            '- Presence of intramedullary rod (metal implant) within femur shaft.\n' +
            '- Bone alignment satisfactory.\n' +
            '- Signs of healing callus formation visible.\n' +
            '- No new fracture or implant loosening.\n\n' +
            'Interpretation:\n' +
            'Healing post-operative femoral fracture with stable fixation.\n\n' +
            'Recommendations:\n' +
            '- Continue physiotherapy and avoid heavy load-bearing.\n' +
            '- Monitor surgical site for pain, redness, or infection.\n' +
            '- Maintain calcium and vitamin D intake.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Repeat X-ray every 4–6 weeks for healing progress.\n' +
            '- Serum calcium + vitamin D levels.\n' +
            '- CT scan if delayed union or non-union suspected.\n' +
            '- Orthopedic review for implant status.',
          symptoms: ['post-op follow-up', 'leg pain (improving)'],
        };
      case 'forearm_xray_radius':
        return {
          predictions: [
            ['Transverse mid-shaft radius fracture (non-displaced)', 0.96],
            ['Ulna intact', 0.9],
          ],
          disease: 'Forearm X-ray — Radius fracture (non-displaced)',
          report:
            '8️⃣ Forearm X-ray (Radius Fracture)\n\n' +
            'Findings:\n' +
            '- Transverse fracture of mid-shaft radius, non-displaced.\n' +
            '- Ulna appears intact.\n' +
            '- Soft tissue shadow normal; no dislocation.\n\n' +
            'Interpretation:\n' +
            'Closed, stable radius fracture with good alignment.\n\n' +
            'Recommendations:\n' +
            '- Immobilize with below-elbow cast/splint.\n' +
            '- Elevate limb to reduce swelling.\n' +
            '- Analgesics/NSAIDs for pain.\n' +
            '- Avoid strain or lifting until bone heals.\n\n' +
            'Suggested Tests / Follow-ups:\n' +
            '- Follow-up X-ray in 2 weeks for alignment.\n' +
            '- Neurovascular check (sensation, pulse).\n' +
            '- MRI if persistent pain → check soft tissue damage.\n' +
            '- Ortho consult if displacement occurs.',
          symptoms: ['forearm pain', 'swelling'],
        };
      default:
        return { predictions: null, disease: 'Mock', report: 'Mock report', symptoms: [] };
    }
  };

  const handleUpload = async () => {
  if (!file) return setError('Please select a file first.');
  if (!selectedImageType) return setError('Please select an image type first.');

  // If mock mode is enabled, bypass API calls and return canned data
  if (mockEnabled) {
    const key = pickMockCase(file?.name, selectedImageType);
    const mock = buildMock(key);

    setProcessedData({
      predictions: mock.predictions,
      report: mock.report,
      disease: mock.disease,
      symptoms: mock.symptoms,
      imagePreview: preview,
      imageType: selectedImageType,
      note: 'Mock mode enabled — results are simulated',
    });

    navigate('/results', {
      state: {
        selectedImageType,
        mockEnabled: !!mockEnabled,
        aiEnabled: localStorage.getItem('aiMode') === 'true',
        heavyModelEnabled: localStorage.getItem('heavyModelMode') === 'true',
        processedData: {
          predictions: mock.predictions,
          report: mock.report,
          disease: mock.disease,
          symptoms: mock.symptoms,
          imagePreview: preview,
          note: 'Mock mode enabled — results are simulated',
        },
      },
    });
    return;
  }

  let predictionEndpoint = '';
  let reportEndpoint = '';

  // Add heavy_mode query parameter if enabled
  const heavyParam = heavyModelEnabled ? '?heavy_mode=true' : '';

  try {
    if (selectedImageType === 'xray') {
      predictionEndpoint = `${BASE_API_URL}/predict/xray/${heavyParam}`;
      reportEndpoint = `${BASE_API_URL}/generate-report/xray/${heavyParam}`;
    } else if (selectedImageType === 'ct_2d') {
      predictionEndpoint = `${BASE_API_URL}/predict/ct/2d/${heavyParam}`;
      reportEndpoint = '';  // Not needed separately — prediction includes report
    } else if (selectedImageType === 'ct_3d') {
      predictionEndpoint = `${BASE_API_URL}/predict/ct/3d/${heavyParam}`;
      reportEndpoint = '';  // Same — prediction handles everything
    } else if (selectedImageType === 'mri_2d') {
      predictionEndpoint = `${BASE_API_URL}/predict/mri/2d/${heavyParam}`;
      reportEndpoint = '';  // Not needed separately — prediction includes report
    } else if (selectedImageType === 'mri_3d') {
      predictionEndpoint = `${BASE_API_URL}/predict/mri/3d/${heavyParam}`;
      reportEndpoint = '';  // Same — prediction handles everything
    } else if (selectedImageType === 'ultrasound') {
      predictionEndpoint = `${BASE_API_URL}/predict/ultrasound/${heavyParam}`;
      reportEndpoint = '';  // Same — prediction handles everything
    } else {
      return setError('Unsupported image type selected.');
    }
  } catch {
    return setError('Invalid image type format.');
  }

  try {
    setUploading(true);
    setError(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    const predictionRes = await axios.post(predictionEndpoint, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        setUploadProgress(percentCompleted);
      },
    });

    // Call report API only for XRAY (others include it in prediction)
    let reportData = {};
    if (selectedImageType === 'xray') {
      const reportRes = await axios.post(reportEndpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      reportData = reportRes.data;
    } else {
      reportData = predictionRes.data;
    }

    setProcessedData({
      predictions: predictionRes.data.predictions || null,
      report: reportData.report,
      disease: reportData.disease,
      symptoms: reportData.symptoms || [],
      imagePreview: preview,
      imageType: selectedImageType
    });

    navigate('/results', {
      state: {
        selectedImageType,
        mockEnabled: !!mockEnabled,
        aiEnabled: localStorage.getItem('aiMode') === 'true',
        heavyModelEnabled: localStorage.getItem('heavyModelMode') === 'true',
        processedData: {
          predictions: predictionRes.data.predictions || null,
          report: reportData.report,
          disease: reportData.disease,
          symptoms: reportData.symptoms || [],
          imagePreview: preview,
        },
      },
    });
  } catch (err) {
    console.error(err);
    setError('An error occurred during upload or analysis. Please try again.');
  } finally {
    setUploading(false);
  }
};


  return (
    <Card className="w-full shadow-md min-h-screen bg-white/80 dark:bg-zinc-950/70 backdrop-blur">
      <CardHeader>
        <CardTitle className="text-xl font-semibold">Upload Medical Image</CardTitle>
      </CardHeader>

      <CardContent>
        <div className="mb-6">
          <ImageTypeSelector
            selectedImageType={selectedImageType}
            setSelectedImageType={setSelectedImageType}
          />
        </div>

        {mockEnabled && (
          <Alert className="mb-4">
            <AlertDescription>
              AI Mode Enabled
            </AlertDescription>
          </Alert>
        )}

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div
          className={`border-2 border-dashed rounded-lg p-6 mb-4 transition-colors duration-300 ${
            preview ? 'border-blue-400 bg-blue-50' : 'border-slate-300 hover:border-blue-400'
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();

            const droppedFile = e.dataTransfer.files[0];
            if (!droppedFile?.type.startsWith('image/')) {
              setError('Please upload an image file.');
              return;
            }

            setFile(droppedFile);
            setError(null);

            const reader = new FileReader();
            reader.onloadend = () => {
              setPreview(reader.result);
            };
            reader.readAsDataURL(droppedFile);
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
            ref={fileInputRef}
          />

          {preview ? (
            <div className="flex flex-col items-center">
              <div className="relative w-full max-w-xs mx-auto">
                <img
                  src={preview}
                  alt="Preview"
                  className="object-cover rounded-md w-full max-h-64"
                />
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute -top-2 -right-2 h-8 w-8 rounded-full shadow-md"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setPreview(null);
                    setError(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <p className="mt-2 text-sm text-slate-500">{file?.name}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center text-center">
              <div className="p-3 rounded-full bg-blue-100 mb-3">
                <Upload className="h-6 w-6 text-blue-600" />
              </div>
              <p className="text-sm font-medium mb-1">Drag and drop your medical image here</p>
              <p className="text-xs text-slate-500 mb-3">or click to browse files</p>
              <p className="text-xs text-slate-400">
                Support for DICOM, JPEG, PNG, and TIFF formats
              </p>
            </div>
          )}
        </div>

        {uploading && (
          <div className="space-y-2 mt-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-500">Uploading & analyzing...</span>
              <span className="text-sm font-medium">{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="h-2" />
          </div>
        )}

        {preview && (
          <div className="mt-6">
            <SegmentedImageViewer imageUrl={preview} imageType={selectedImageType} />
          </div>
        )}
      </CardContent>

      <CardFooter className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Image className="h-4 w-4 text-slate-500" />
          <span className="text-sm text-slate-500">
            {selectedImageType
              ? selectedImageType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
              : 'Select image type first'}
          </span>
        </div>
        <Button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 transition-all duration-300"
        >
          {uploading ? (
            <span className="flex items-center gap-2">
              Processing <span className="animate-pulse">...</span>
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <Check className="h-4 w-4" /> Upload & Analyze
            </span>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default UploadPage;
