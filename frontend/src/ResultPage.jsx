import React, { useEffect, useState } from 'react';
import ReportCard from './ReportCard';
import { useParams, useLocation } from 'react-router-dom'; // added useLocation

const BASE_API_URL = 'http://127.0.0.1:8000';

const ResultsPage = () => {
  const { cleanType } = useParams(); // e.g., 'xray', 'ct', etc.
  const location = useLocation();

  // Try to get the passed state from navigation:
  const { processedData, selectedImageType, mockEnabled, aiEnabled: aiEnabledState, heavyModelEnabled: heavyModelEnabledState } = location.state || {};

  const [reportData, setReportData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [aiResult, setAiResult] = useState(null);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [heavyModelEnabled, setHeavyModelEnabled] = useState(false);

  useEffect(() => {
    // Prefer state flag, fallback to localStorage for resilience
    const ls = localStorage.getItem('aiMode') === 'true';
    setAiEnabled(Boolean(aiEnabledState ?? ls));
    const heavyLs = localStorage.getItem('heavyModelMode') === 'true';
    setHeavyModelEnabled(Boolean(heavyModelEnabledState ?? heavyLs));
  }, [aiEnabledState, heavyModelEnabledState]);

  const toBaseModality = (t) => {
    if (!t) return 'xray';
    const s = String(t).toLowerCase();
    if (s.startsWith('ct')) return 'ct';
    if (s.startsWith('mri')) return 'mri';
    if (s.includes('ultra')) return 'ultrasound';
    return s;
  };

  const dataURLtoBlob = (dataurl) => {
    if (!dataurl) return null;
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new Blob([u8arr], { type: mime });
  };

  const runAiAnalysis = async () => {
    try {
      setAiError(null);
      setAiLoading(true);
      const blob = dataURLtoBlob(processedData?.imagePreview);
      if (!blob) throw new Error('No preview image available');
      const form = new FormData();
      form.append('image_type', toBaseModality(selectedImageType));
      form.append('file', blob, 'upload.png');
      const llmMode = localStorage.getItem('llmMode') || 'auto';
      const res = await fetch(`${BASE_API_URL}/ai/analyze/?llm_mode=${encodeURIComponent(llmMode)}`, { method: 'POST', body: form });
      if (!res.ok) throw new Error('AI analysis failed');
      const json = await res.json();
      setAiResult(json);
    } catch (e) {
      setAiError(e.message || 'Failed to run AI analysis');
    } finally {
      setAiLoading(false);
    }
  };

  useEffect(() => {
    // If we have processedData from navigation, use it directly
    if (processedData) {
      // Construct your reportData using processedData from UploadPage
      const predictionData = processedData.predictions || [];
      const reportText = processedData.report || '';
      const disease = processedData.disease || '';
      const symptoms = processedData.symptoms || [];

      // Find the top prediction for confidence and diagnosis
      const sorted = Array.isArray(predictionData) && predictionData.length
        ? [...predictionData].sort((a, b) => b[1] - a[1])
        : [];

      const topK = sorted.slice(0, 3);
      const topSymptoms = symptoms.length ? symptoms : topK.map(([cond]) => cond);
      const [bestCond, bestScore] = sorted.length ? sorted[0] : [disease, 1];

      const specialtyMap = {
        Pneumonia: 'Pulmonologist',
        'Pleural Effusion': 'Pulmonologist',
        'Congestive Heart Failure': 'Cardiologist',
        Fracture: 'Orthopedic Surgeon',
        Glioma: 'Neurologist',
        Meningioma: 'Neurologist',
      };

      const specialty = specialtyMap[bestCond] || (disease ? 'Specialist' : 'General Physician');

      const rawConf = sorted.length ? Math.round((bestScore || 0) * 100) : 72;
      const confidence = Math.max(35, Math.min(95, rawConf));

      const formattedReport = {
        symptoms: topSymptoms,
        diagnosis: disease || bestCond,
        confidence,
        recommendations: [],
        suggested_tests: [],
        specialty,
        timestamp: new Date().toISOString(),
        heavyModelUsed: heavyModelEnabled,
        report: reportText,
      };

      setReportData(formattedReport);
      setLoading(false);
      return;
    }

    // Fallback: if no processedData, fetch from API as before
    if (!cleanType) {
      setError('No image type specified and no data passed.');
      setLoading(false);
      return;
    }

    const fetchReport = async () => {
      try {
        const [predictionRes, reportRes] = await Promise.all([
          fetch(`${BASE_API_URL}/predict/${cleanType}/`),
          fetch(`${BASE_API_URL}/generate-report/${cleanType}/`),
        ]);

        if (!predictionRes.ok || !reportRes.ok) {
          throw new Error('One of the API calls failed');
        }

        const predictionData = await predictionRes.json();
        const reportDataRaw = await reportRes.json();

        const sorted = predictionData.predictions.sort((a, b) => b[1] - a[1]);
        const topK = sorted.slice(0, 3);
        const symptoms = topK.map(([cond]) => cond);
        const [bestCond, bestScore] = sorted[0];

        const specialtyMap = {
          Pneumonia: 'Pulmonologist',
          'Pleural Effusion': 'Pulmonologist',
          'Congestive Heart Failure': 'Cardiologist',
          Fracture: 'Orthopedic Surgeon',
          Glioma: 'Neurologist',
          Meningioma: 'Neurologist',
        };

        const specialty = specialtyMap[bestCond] || (reportDataRaw.disease ? 'Specialist' : 'General Physician');

        const rawConf = Math.round((sorted?.[0]?.[1] || 0) * 100);
        const confidence = Math.max(35, Math.min(95, rawConf || 72));

        const formattedReport = {
          symptoms,
          diagnosis: reportDataRaw.disease || bestCond,
          confidence,
          recommendations: [],
          suggested_tests: [],
          specialty,
          timestamp: new Date().toISOString(),
          report: reportDataRaw.report || '',
        };

        setReportData(formattedReport);
      } catch (err) {
        console.error(err);
        setError('Failed to load report. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [cleanType, processedData]);

  if (loading) {
    return (
      <div className="p-6 text-center text-slate-500 min-h-screen">Loading report...</div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-2xl text-red-600 opacity-60 min-h-screen">
        {error} 😢
      </div>
    );
  }

  return (
    <div className="p-6 min-h-screen space-y-6 max-w-4xl mx-auto">
      <div id="report-content">
        <ReportCard report={reportData} />
      </div>
      {aiEnabled && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 p-4 bg-white/80 dark:bg-zinc-900/70 backdrop-blur">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">AI Option (Gemini 2.5 Pro)</h3>
            <button
              onClick={runAiAnalysis}
              disabled={aiLoading}
              className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {aiLoading ? 'Analyzing…' : 'Generate AI Analysis'}
            </button>
          </div>
          {aiError && (
            <p className="mt-2 text-sm text-red-600">{aiError}</p>
          )}
          {aiResult && (
            <div className="mt-4 space-y-5">
              {/* Provider badge */}
              {(aiResult.model || aiResult.provider) && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Provider</span>
                  <span className="text-xs font-medium rounded-full border border-slate-200 dark:border-slate-700 px-2 py-1 bg-white/70 dark:bg-zinc-900/60">
                    {aiResult.model || aiResult.provider}
                  </span>
                </div>
              )}
              {aiResult.condition && (
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Condition Detected</h4>
                    <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">{aiResult.condition}</p>
                  </div>
                  {typeof aiResult.confidence === 'number' && (
                    <div className="text-sm rounded-md px-2 py-1 border border-slate-200 dark:border-slate-700">
                      AI Confidence: <span className="font-medium">{aiResult.confidence}%</span>
                    </div>
                  )}
                </div>
              )}

              {aiResult.findings && (
                <div>
                  <h4 className="font-medium mb-1">Findings</h4>
                  <p className="whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">{aiResult.findings}</p>
                </div>
              )}

              {aiResult.impression && (
                <div>
                  <h4 className="font-medium mb-1">Impression</h4>
                  <p className="whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">{aiResult.impression}</p>
                </div>
              )}

              {Array.isArray(aiResult.suggested_tests) && aiResult.suggested_tests.length > 0 && (
                <div>
                  <h4 className="font-medium mb-1">Suggested Diagnostic Tests</h4>
                  <ul className="list-disc ml-5 space-y-1 text-sm text-slate-700 dark:text-slate-300">
                    {aiResult.suggested_tests.map((t, idx) => (<li key={idx}>{t}</li>))}
                  </ul>
                </div>
              )}

              {Array.isArray(aiResult.recommendations) && aiResult.recommendations.length > 0 && (
                <div>
                  <h4 className="font-medium mb-1">Clinical Recommendations</h4>
                  <ul className="list-none space-y-1 text-sm text-slate-700 dark:text-slate-300">
                    {aiResult.recommendations.map((line, idx) => (
                      <li key={idx}>{line}</li>
                    ))}
                  </ul>
                </div>
              )}

              {aiResult.explanation && (
                <div>
                  <h4 className="font-medium mb-1">Full Diagnostic Explanation</h4>
                  <p className="whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">{aiResult.explanation}</p>
                </div>
              )}

              {!aiResult.findings && !aiResult.impression && !aiResult.explanation && aiResult.analysis && (
                <div>
                  <h4 className="font-medium mb-1">Medical Analysis</h4>
                  <p className="whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">{aiResult.analysis}</p>
                </div>
              )}

              {/* Download JSON */}
              <div className="pt-2">
                <button
                  onClick={() => {
                    try {
                      const blob = new Blob([JSON.stringify(aiResult, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `ai_analysis_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
                      document.body.appendChild(a);
                      a.click();
                      a.remove();
                      URL.revokeObjectURL(url);
                    } catch {}
                  }}
                  className="px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 text-sm bg-white/80 dark:bg-zinc-900/70 hover:bg-white"
                >
                  Download JSON
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ResultsPage;
