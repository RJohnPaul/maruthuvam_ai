import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Input } from './components/ui/input';
import { Button } from './components/ui/button';
import { Card, CardContent } from './components/ui/card';

const BASE_API_URL = 'http://127.0.0.1:8000';

const ResultChatPage = () => {
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: "Hi! I'm your report assistant. How can I help you with your diagnosis?",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [provider, setProvider] = useState(null);
  const [model, setModel] = useState(null);
  const [availableContexts, setAvailableContexts] = useState([]);
  const [selectedModality, setSelectedModality] = useState('');
  const [selectedContext, setSelectedContext] = useState(null);

  // Discover latest reports from any modality to ground the chat
  useEffect(() => {
    const fetchContexts = async () => {
      const endpoints = [
        { key: 'xray', url: `${BASE_API_URL}/get-latest-report/xray/` },
        { key: 'ct2d', url: `${BASE_API_URL}/get-latest-report/ct2d/` },
        { key: 'ct3d', url: `${BASE_API_URL}/get-latest-report/ct3d/` },
        { key: 'mri3d', url: `${BASE_API_URL}/get-latest-report/mri3d/` },
        { key: 'ultrasound', url: `${BASE_API_URL}/get-latest-report/ultrasound/` },
      ];
      const results = await Promise.allSettled(endpoints.map(e => fetch(e.url)));
      const ok = [];
      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        if (r.status === 'fulfilled' && r.value.ok) {
          const data = await r.value.json();
          ok.push({ key: endpoints[i].key, data });
        }
      }
      setAvailableContexts(ok);
      if (ok.length) {
        setSelectedModality(ok[0].key);
        setSelectedContext(ok[0].data);
      }
    };
    fetchContexts().catch(() => {});
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const llmMode = localStorage.getItem('llmMode') || 'auto';
      const payload = {
        modality: selectedModality || null,
        report_context: selectedContext || null,
        messages: [...messages, userMessage],
      };
      const res = await axios.post(`${BASE_API_URL}/ai/chat/?llm_mode=${encodeURIComponent(llmMode)}`, payload);
      const aiReply = res.data.reply || 'Sorry, I could not understand that.';
      setProvider(res.data.provider || null);
      setModel(res.data.model || null);
      setMessages((prev) => [...prev, { role: 'ai', text: aiReply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: 'ai', text: 'There was an error fetching a response. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-6 flex flex-col items-center">
      <Card className="w-full max-w-3xl flex flex-col flex-grow border border-border shadow-lg">
        <CardContent className="p-4 space-y-4 flex flex-col flex-grow">
          <h2 className="text-2xl font-semibold text-center">Report Chat Assistant</h2>

          {/* Context selector and provider badge */}
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-600 dark:text-slate-400">Context</span>
              <select
                value={selectedModality}
                onChange={(e) => {
                  const key = e.target.value;
                  setSelectedModality(key);
                  const found = availableContexts.find(c => c.key === key);
                  setSelectedContext(found ? found.data : null);
                }}
                className="text-sm rounded-md border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-900/70 px-2 py-1 focus:outline-none"
              >
                {availableContexts.length === 0 && <option value="">None</option>}
                {availableContexts.map((c) => (
                  <option key={c.key} value={c.key}>{c.key}</option>
                ))}
              </select>
            </div>
            {(provider || model) && (
              <div className="text-xs rounded-full border border-slate-200 dark:border-slate-700 px-2 py-1 bg-white/70 dark:bg-zinc-900/60">
                {provider}{model ? ` · ${model}` : ''}
              </div>
            )}
          </div>

          <div className="flex-grow overflow-y-auto max-h-[60vh] space-y-3 p-2 border rounded-md bg-muted/30">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-md text-sm max-w-[80%] ${
                  msg.role === 'ai'
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 font-semibold self-start'
                    : 'bg-zinc-800 dark:bg-white text-white dark:text-black font-semibold self-end ml-auto'
                }`}
              >
                {msg.text}
              </div>
            ))}
          </div>

          <form
            className="flex gap-2 pt-4"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <Input
              className="flex-grow"
              placeholder="Ask about your diagnosis..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
            />
            <Button type="submit" disabled={loading}>
              {loading ? 'Thinking...' : 'Send'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResultChatPage;
