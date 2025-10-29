import React, { useEffect, useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import LandingPage from './LandingPage';
import UploadPage from './UploadPage';
import ResultPage from './ResultPage';
import Header from './components/Header';
import Footer from './components/Footer';
import Contact from './Contact';
import ResultChatPage from './ResultChatPage';
import DoctorSearchPage from './DoctorSearchPage';
import MockToggle from './components/MockToggle';
import DarkVeil from './DarkVeil';

function App() {
  // Global state for selected image type and processed data
  const [selectedImageType, setSelectedImageType] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [mockEnabled, setMockEnabled] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [heavyModelEnabled, setHeavyModelEnabled] = useState(false);
  const location = useLocation();
  const isUploadRoute = location.pathname.startsWith('/upload');

  // Hydrate toggles from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('mockMode');
    if (saved === 'true') setMockEnabled(true);
    const aiSaved = localStorage.getItem('aiMode');
    if (aiSaved === 'true') setAiEnabled(true);
    const heavySaved = localStorage.getItem('heavyModelMode');
    if (heavySaved === 'true') setHeavyModelEnabled(true);
  }, []);

  const handleMockChange = (value) => {
    setMockEnabled(Boolean(value));
    localStorage.setItem('mockMode', Boolean(value).toString());
  };

  const handleAiChange = (value) => {
    setAiEnabled(Boolean(value));
    localStorage.setItem('aiMode', Boolean(value).toString());
  };

  const handleHeavyModelChange = (value) => {
    setHeavyModelEnabled(Boolean(value));
    localStorage.setItem('heavyModelMode', Boolean(value).toString());
  };

  return (
    <div className="relative flex flex-col min-h-screen">
      {/* Global Dark Veil background (hidden on Upload page) */}
      {!isUploadRoute && (
        <div className="pointer-events-none absolute inset-0 -z-10">
          <DarkVeil hueShift={8} noiseIntensity={0.015} scanlineIntensity={0.025} scanlineFrequency={7.5} speed={0.4} warpAmount={0.045} />
        </div>
      )}
      <Header />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route 
            path="/upload" 
            element={
              <UploadPage 
                selectedImageType={selectedImageType} 
                setSelectedImageType={setSelectedImageType} 
                setProcessedData={setProcessedData}
                mockEnabled={mockEnabled}
                aiEnabled={aiEnabled}
                heavyModelEnabled={heavyModelEnabled}
              />
            } 
          />
          <Route 
            path="/results" 
            element={
              <ResultPage />
            } 
          />
          <Route path="/resultchat" element={<ResultChatPage />} />
          <Route path="/search-doctor" element={<DoctorSearchPage />} />
          <Route path='/contact' element={<Contact/>} />
        </Routes>
      </main>
      <Footer />
      {/* Floating toggles */}
      <MockToggle 
        enabled={mockEnabled} 
        onChange={handleMockChange} 
        aiEnabled={aiEnabled} 
        onAiChange={handleAiChange}
        heavyModelEnabled={heavyModelEnabled}
        onHeavyModelChange={handleHeavyModelChange}
      />
    </div>
  );
}

export default App;
