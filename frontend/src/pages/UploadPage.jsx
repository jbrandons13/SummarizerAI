import React, { useState } from 'react';
import FileDropzone from '../components/FileDropzone';
import SettingsPanel from '../components/SettingsPanel';
import PipelineOverview from '../components/PipelineOverview';
import { submitVideo } from '../api';

export default function UploadPage({ onStartJob, onError }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState({
    retrieval_method: 'all',
    style: 'informative',
    subtitles: 'none',
    tts_backend: 'kokoro',
    llm_backend: 'local'
  });

  const handleGenerate = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const { job_id } = await submitVideo(file, settings);
      onStartJob(job_id);
    } catch (err) {
      onError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col space-y-1">
        <h1 className="text-2xl font-bold text-gray-900">New Summarization</h1>
        <p className="text-sm text-gray-500">Upload a video and configure the pipeline settings for the best results.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <FileDropzone onFileSelect={setFile} selectedFile={file} />
          
          <div className="card bg-blue-50 border-blue-100 flex items-start space-x-3">
            <div className="text-blue-500 mt-0.5">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="text-xs text-blue-700 leading-relaxed">
              <strong>Info:</strong> Default mode is <strong>Compare all 3</strong> (Ablation Mode). The system will run all three methods so you can compare the results in the Dashboard.
            </div>
          </div>

          <PipelineOverview retrievalMethod={settings.retrieval_method} />
        </div>

        <div className="space-y-6">
          <SettingsPanel settings={settings} onChange={setSettings} />
          
          <button 
            disabled={!file || loading}
            onClick={handleGenerate}
            className="w-full btn-premium py-4 rounded-xl font-bold flex items-center justify-center space-x-2 group"
          >
            {loading ? (
              <div className="flex items-center space-x-3">
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="tracking-wide">AI PREPARING ENGINE...</span>
              </div>
            ) : (
              <>
                <svg className="w-5 h-5 transform group-hover:rotate-12 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="tracking-tight uppercase">Generate Magic Summary</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
