import React, { useState, useEffect } from 'react';
import MetricCard from '../components/MetricCard';
import ComparisonView from '../components/ComparisonView';
import ScriptTimeline from '../components/ScriptTimeline';
import { getResult } from '../api';

export default function ResultsPage({ jobId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadResults = async () => {
      try {
        const result = await getResult(jobId);
        setData(result);
      } catch (err) {
        console.error("Failed to load result", err);
      } finally {
        setLoading(false);
      }
    };
    loadResults();
  }, [jobId]);

  const handleSeek = (time) => {
    // We'd ideally find all video elements and seek them
    const videos = document.querySelectorAll('video');
    videos.forEach(v => {
      v.currentTime = time;
      v.play();
    });
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <p className="text-gray-500 font-medium">Finalizing results...</p>
      </div>
    );
  }

  if (!data) return <div>Failed to load results.</div>;

  return (
    <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-700">
      <div className="flex justify-between items-center">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-gray-900">Summarization Complete</h1>
          <p className="text-sm text-gray-500">Multimodal pipeline finished. Review findings below.</p>
          {data.config && (
            <div className="flex gap-2 mt-2">
              <span className="badge badge-blue">LLM: {data.config.llm_backend}</span>
              <span className="badge badge-purple">TTS: {data.config.tts_backend}</span>
              <span className="badge badge-green">Retrieval: {data.config.retrieval_method}</span>
            </div>
          )}
        </div>
        <div className="flex space-x-2">
           <button className="btn btn-primary">
             Download Best Arm
           </button>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard 
          label="Original Length" 
          value={`${data.original_duration.toFixed(1)}s`} 
        />
        <MetricCard 
          label="Summary Length" 
          value={`${data.summary_duration.toFixed(1)}s`} 
        />
        <MetricCard 
          label="Compression" 
          value={`${((1 - (data.summary_duration / data.original_duration)) * 100).toFixed(0)}%`} 
          subvalue="Reduction in duration"
        />
      </div>

      {/* Comparison View */}
      <ComparisonView 
        method={data.method || 'siglip_direct'} 
        outputs={data.outputs} 
        onSeekAll={handleSeek}
      />

      {/* Script Timeline */}
      <ScriptTimeline 
        script={data.summary_script} 
        onSeek={handleSeek}
      />
    </div>
  );
}
