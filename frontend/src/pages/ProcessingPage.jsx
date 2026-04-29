import React, { useState, useEffect } from 'react';
import PhaseProgress from '../components/PhaseProgress';
import { getStatus, connectProgress, cancelJob } from '../api';

const PIPELINE_PHASES = [
  { id: 1, name: 'Precision Transcription', key: 'phase1' },
  { id: 2, name: 'LLM Summarization', key: 'phase2' },
  { id: 3, name: 'Neural Voiceover', key: 'phase3' },
  { id: 4, name: 'Semantic Retrieval', key: 'phase4' },
  { id: 5, name: 'Final Assembly', key: 'phase5' }
];

export default function ProcessingPage({ jobId, onComplete, onError }) {
  const [elapsed, setElapsed] = useState(0);
  const [config, setConfig] = useState(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [phaseStates, setPhaseStates] = useState(PIPELINE_PHASES.map(p => ({
    ...p,
    status: 'waiting',
    progress: 0,
    detail: '',
    vram: null,
    duration: null
  })));

  const updatePhaseStates = (data) => {
    if (data.config && !config) {
      setConfig(data.config);
    }
    
    // Sync elapsed time from server
    if (data.elapsed_seconds !== undefined) {
      setElapsed(data.elapsed_seconds);
    }

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

  useEffect(() => {
    // 1. Initial poll and local ticker (for smooth appearance)
    const ticker = setInterval(() => {
      setElapsed(e => e + 1);
    }, 1000);

    const performPoll = async () => {
      try {
        const status = await getStatus(jobId);
        if (status.status === 'completed') {
          onComplete();
          return true;
        } else if (status.status === 'failed') {
          onError(status.error);
          return true;
        } else if (status.status === 'cancelled') {
          onError("Job was cancelled by user.");
          return true;
        } else {
          updatePhaseStates(status);
        }
      } catch (e) {
        console.error("Polling error", e);
      }
      return false;
    };

    // 2. Main polling interval
    const pollId = setInterval(async () => {
      const finished = await performPoll();
      if (finished) {
        clearInterval(ticker);
        clearInterval(pollId);
      }
    }, 3000);

    // 3. SECRECY: Immediate refresh when tab becomes visible
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        performPoll();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // 4. WebSocket (optional enhancement)
    const ws = connectProgress(jobId, (data) => {
      if (data.status === 'completed') {
        onComplete();
      } else if (data.status === 'failed') {
        onError(data.error || 'Pipeline failed');
      } else if (data.status === 'cancelled') {
        onError("Job was cancelled by user.");
      } else {
        updatePhaseStates(data);
      }
    });

    // 5. Browser Close/Refresh Guard
    const handleBeforeUnload = (e) => {
      e.preventDefault();
      e.returnValue = ''; // Required for Chrome
    };
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      clearInterval(ticker);
      clearInterval(pollId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('beforeunload', handleBeforeUnload);
      ws.close();
    };
  }, [jobId]);

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
          <button 
            disabled={isCancelling}
            onClick={async () => {
              if (window.confirm("Stop this job? Current progress will be lost.")) {
                try {
                  setIsCancelling(true);
                  await cancelJob(jobId);
                } catch (e) {
                  console.error("Cancel failed", e);
                  setIsCancelling(false);
                  alert("Failed to send cancel request. Please try again.");
                }
              }
            }}
            className={`mt-2 text-[10px] font-bold uppercase tracking-tighter transition-colors flex items-center justify-end space-x-1 ${
              isCancelling ? 'text-gray-400 cursor-not-allowed' : 'text-red-400 hover:text-red-500'
            }`}
          >
            <div className={`w-2 h-2 rounded-full ${isCancelling ? 'bg-gray-300 animate-pulse' : 'bg-red-400'}`}></div>
            <span>{isCancelling ? 'Stopping Processing...' : 'Stop Processing'}</span>
          </button>
        </div>
      </div>

      <div className="card space-y-8 py-10">
        <div className="max-w-xl mx-auto space-y-10">
          {/* Config Badges */}
          {config && (
            <div className="flex flex-wrap gap-2 justify-center pb-6 border-b border-gray-100">
              <div className="px-3 py-1 bg-blue-50 text-blue-700 text-xs font-semibold rounded-full border border-blue-100">
                LLM: <span className="uppercase">{config.llm_backend || 'local'}</span>
              </div>
              <div className="px-3 py-1 bg-purple-50 text-purple-700 text-xs font-semibold rounded-full border border-purple-100">
                TTS: <span className="uppercase">{config.tts_backend || 'kokoro'}</span>
              </div>
              <div className="px-3 py-1 bg-green-50 text-green-700 text-xs font-semibold rounded-full border border-green-100">
                Method: <span className="uppercase">{config.retrieval_method?.replace('_', ' ') || 'all'}</span>
              </div>
              <div className="px-3 py-1 bg-orange-50 text-orange-700 text-xs font-semibold rounded-full border border-orange-100">
                Style: <span className="uppercase">{config.style || 'informative'}</span>
              </div>
              <div className="px-3 py-1 bg-teal-50 text-teal-700 text-xs font-semibold rounded-full border border-teal-100">
                Subtitles: <span className="uppercase">{config.subtitles !== 'none' ? 'YES' : 'NO'}</span>
              </div>
            </div>
          )}

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
