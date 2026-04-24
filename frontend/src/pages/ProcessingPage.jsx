import React, { useState, useEffect } from 'react';
import PhaseProgress from '../components/PhaseProgress';
import { getStatus, connectProgress } from '../api';

const PIPELINE_PHASES = [
  { id: 1, name: 'Precision Transcription', key: 'phase1' },
  { id: 2, name: 'LLM Summarization', key: 'phase2' },
  { id: 3, name: 'Neural Voiceover', key: 'phase3' },
  { id: 4, name: 'Semantic Retrieval', key: 'phase4' },
  { id: 5, name: 'Final Assembly', key: 'phase5' }
];

export default function ProcessingPage({ jobId, onComplete, onError }) {
  const [elapsed, setElapsed] = useState(0);
  const [phaseStates, setPhaseStates] = useState(PIPELINE_PHASES.map(p => ({
    ...p,
    status: 'waiting',
    progress: 0,
    detail: '',
    vram: null,
    duration: null
  })));

  useEffect(() => {
    let timer = setInterval(() => setElapsed(e => e + 1), 1000);
    
    // Connect WebSocket
    const ws = connectProgress(jobId, (data) => {
      // Update states based on socket data
      // Data expected: { current_phase, progress_pct, phase_details, status, ... }
      if (data.status === 'completed') {
        clearInterval(timer);
        onComplete();
      } else if (data.status === 'failed') {
        clearInterval(timer);
        onError(data.error || 'Pipeline failed during execution');
      } else {
        updatePhaseStates(data);
      }
    }, (err) => {
      console.warn("WS error, falling back to polling", err);
    });

    // Polling fallback
    const poll = setInterval(async () => {
      try {
        const status = await getStatus(jobId);
        if (status.status === 'completed') {
          clearInterval(timer);
          clearInterval(poll);
          onComplete();
        } else if (status.status === 'failed') {
          clearInterval(timer);
          clearInterval(poll);
          onError(status.error);
        } else {
          updatePhaseStates(status);
        }
      } catch (e) {
        console.error("Polling error", e);
      }
    }, 3000);

    return () => {
      clearInterval(timer);
      clearInterval(poll);
      ws.close();
    };
  }, [jobId]);

  const updatePhaseStates = (data) => {
    setPhaseStates(prev => prev.map((p, idx) => {
      const phaseNum = idx + 1;
      if (phaseNum < data.current_phase) {
        return { ...p, status: 'done', progress: 100 };
      } else if (phaseNum === data.current_phase) {
        return { 
          ...p, 
          status: 'in_progress', 
          progress: data.progress_pct, 
          detail: data.phase_details,
          vram: data.vram_peak,
          duration: data.elapsed_seconds
        };
      } else {
        return { ...p, status: 'waiting', progress: 0 };
      }
    }));
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex justify-between items-end">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-gray-900">Processing Video</h1>
          <p className="text-sm text-gray-500">Pipeline execution in progress. This may take a few minutes.</p>
        </div>
        <div className="text-right">
          <div className="text-sm font-medium text-gray-400 uppercase tracking-widest">Elapsed Time</div>
          <div className="text-2xl font-mono text-blue-600 font-bold">
            {Math.floor(elapsed / 60)}:{(elapsed % 60).toString().padStart(2, '0')}
          </div>
        </div>
      </div>

      <div className="card space-y-8 py-10">
        <div className="max-w-xl mx-auto space-y-10">
          {phaseStates.map((phase, idx) => (
            <PhaseProgress 
              key={phase.id} 
              phase={phase} 
              index={idx} 
              totalPhases={PIPELINE_PHASES.length} 
            />
          ))}
        </div>
      </div>

      <div className="flex flex-col items-center justify-center space-y-4">
        <p className="text-xs text-gray-400 max-w-sm text-center">
          The models are currently running on the server GPU. 
          VRAM peaks are recorded for each phase to ensure optimal resource allocation.
        </p>
        <div className="flex space-x-1">
           {[...Array(3)].map((_, i) => (
             <div key={i} className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.2}s` }} />
           ))}
        </div>
      </div>
    </div>
  );
}
