import React, { useEffect, useState } from 'react';
import { Switch } from './ui/switch';
import { Badge } from './ui/badge';

/**
 * Floating bottom-left quick controls for AI, Heavy Model, and Mock modes.
 * - AI Mode persists to localStorage key "aiMode"
 * - Heavy Model Mode persists to localStorage key "heavyModelMode"
 * - Mock Mode persists to localStorage key "mockMode"
 */
const MockToggle = ({ enabled, onChange, aiEnabled = false, onAiChange = () => {}, heavyModelEnabled = false, onHeavyModelChange = () => {} }) => {
  const [llmMode, setLlmMode] = useState('auto');

  useEffect(() => {
    const saved = localStorage.getItem('llmMode');
    if (saved) setLlmMode(saved);
  }, []);

  const handleLlmChange = (e) => {
    const val = e.target.value;
    setLlmMode(val);
    localStorage.setItem('llmMode', val);
  };

  return (
    <div className="fixed bottom-4 left-4 z-50 select-none">
      <div className="flex flex-col gap-3 rounded-lg bg-white/90 dark:bg-slate-900/90 backdrop-blur px-5 py-3 shadow-md border border-slate-200 dark:border-slate-800">
        {/* AI toggle */}
        <div className="flex items-center gap-2">
          <Switch id="ai-mode" checked={aiEnabled} onCheckedChange={onAiChange} />
          <label htmlFor="ai-mode" className="text-sm font-medium text-slate-700 dark:text-slate-200 min-w-[3.5rem]">
            AI
          </label>
          <Badge variant={aiEnabled ? 'default' : 'secondary'}>{aiEnabled ? 'ON' : 'OFF'}</Badge>
        </div>
        
        {/* Heavy Model toggle */}
        <div className="flex items-center gap-2">
          <Switch id="heavy-mode" checked={heavyModelEnabled} onCheckedChange={onHeavyModelChange} />
          <label htmlFor="heavy-mode" className="text-sm font-medium text-slate-700 dark:text-slate-200 min-w-[3.5rem]">
            Heavy
          </label>
          <Badge variant={heavyModelEnabled ? 'destructive' : 'secondary'}>{heavyModelEnabled ? 'ON' : 'OFF'}</Badge>
        </div>
        
        {/* LLM mode selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="llm-mode-select" className="text-sm font-medium text-slate-700 dark:text-slate-200 min-w-[3.5rem]">
            LLM
          </label>
          <select
            id="llm-mode-select"
            value={llmMode}
            onChange={handleLlmChange}
            className="text-sm rounded-md border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-900/70 px-2 py-1 focus:outline-none"
          >
            <option value="auto">auto</option>
            <option value="text-only">text-only</option>
            <option value="offline">offline</option>
          </select>
          <Badge variant="secondary" className="capitalize">{llmMode}</Badge>
        </div>

        {/* Mock toggle */}
        <div className="flex items-center gap-2">
          <Switch id="mock-mode" checked={enabled} onCheckedChange={onChange} />
          <label htmlFor="mock-mode" className="text-sm font-medium text-slate-700 dark:text-slate-200 min-w-[3.5rem]">
            Modelv1
          </label>
          <Badge variant={enabled ? 'default' : 'secondary'}>{enabled ? 'ON' : 'OFF'}</Badge>
        </div>
      </div>
    </div>
  );
};

export default MockToggle;
