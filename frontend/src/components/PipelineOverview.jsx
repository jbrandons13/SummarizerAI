import React, { useState } from 'react';

const Step = ({ icon, label, subtitle, tech }) => (
  <div className="flex flex-col items-center text-center gap-1.5 min-w-0 flex-1">
    <div className="w-12 h-12 rounded-xl bg-gray-50 border border-gray-100 flex items-center justify-center text-gray-400">
      {icon}
    </div>
    <span className="text-xs font-medium text-gray-800">{label}</span>
    <span className="text-[11px] text-gray-400 leading-tight">{subtitle}</span>
    <span className="text-[10px] bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">{tech}</span>
  </div>
);

const Arrow = () => (
  <>
    {/* Desktop Arrow */}
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="text-gray-300 hidden md:block flex-shrink-0" strokeWidth="1.5">
      <path d="M5 12h14"/>
      <path d="M12 5l7 7-7 7"/>
    </svg>
    {/* Mobile Arrow */}
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="text-gray-200 block md:hidden flex-shrink-0" strokeWidth="1.5">
      <path d="M12 5v14"/>
      <path d="M19 12l-7 7-7-7"/>
    </svg>
  </>
);

export default function PipelineOverview({ retrievalMethod }) {
  const [isOpen, setIsOpen] = useState(false);

  const getRetrievalTech = () => {
    switch (retrievalMethod) {
      case 'random': return 'Random';
      case 'caption_cosine': return 'Qwen-VL';
      case 'siglip_direct': return 'SigLIP 2';
      case 'all': return '3 Methods';
      default: return 'SigLIP 2';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden transition-all duration-300">
      {/* Header */}
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <span className="text-sm font-medium text-gray-700">How it works</span>
        <svg 
          className={`w-4 h-4 text-gray-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content */}
      <div 
        className={`transition-all duration-500 ease-in-out ${isOpen ? 'max-h-[800px] opacity-100' : 'max-h-0 opacity-0'}`}
      >
        <div className="p-5 pt-0 border-t border-gray-50">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 py-4">
            {/* Step 1 */}
            <Step 
              label="Transcribe"
              subtitle="Speech → text with timestamps"
              tech="WhisperX"
              icon={(
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                  <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/>
                  <path d="M19 10v2a7 7 0 01-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              )}
            />
            
            <Arrow />

            {/* Step 2 */}
            <Step 
              label="Summarize"
              subtitle="AI writes narration script"
              tech="Llama 3.3"
              icon={(
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <path d="M9 15l2 2 4-4"/>
                </svg>
              )}
            />

            <Arrow />

            {/* Step 3 */}
            <Step 
              label="Voiceover"
              subtitle="Text → natural speech audio"
              tech="Kokoro TTS"
              icon={(
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                  <path d="M11 5L6 9H2v6h4l5 4V5z"/>
                  <path d="M19.07 4.93a10 10 0 010 14.14"/>
                  <path d="M15.54 8.46a5 5 0 010 7.07"/>
                </svg>
              )}
            />

            <Arrow />

            {/* Step 4 */}
            <Step 
              label="Match visuals"
              subtitle="Find best B-roll per sentence"
              tech={getRetrievalTech()}
              icon={(
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                  <rect x="3" y="3" width="7" height="7"/>
                  <rect x="14" y="3" width="7" height="7"/>
                  <rect x="3" y="14" width="7" height="7"/>
                  <rect x="14" y="14" width="7" height="7"/>
                  <circle cx="18.5" cy="18.5" r="2.5"/>
                  <line x1="21" y1="21" x2="23" y2="23"/>
                </svg>
              )}
            />

            <Arrow />

            {/* Step 5 */}
            <Step 
              label="Assemble"
              subtitle="Cut, stitch & render final video"
              tech="FFmpeg"
              icon={(
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                  <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
                  <line x1="7" y1="2" x2="7" y2="22"/>
                  <line x1="17" y1="2" x2="17" y2="22"/>
                  <line x1="2" y1="12" x2="22" y2="12"/>
                  <line x1="2" y1="7" x2="7" y2="7"/>
                  <line x1="2" y1="17" x2="7" y2="17"/>
                  <line x1="17" y1="7" x2="22" y2="7"/>
                  <line x1="17" y1="17" x2="22" y2="17"/>
                </svg>
              )}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
