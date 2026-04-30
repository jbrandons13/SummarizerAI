import React from 'react';

export default function SettingsPanel({ settings, onChange }) {
  const handleChange = (name, value) => {
    onChange({ ...settings, [name]: value });
  };

  return (
    <div className="card space-y-5">
      <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3">Pipeline Settings</div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Retrieval Method */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-gray-500">Retrieval method</label>
          <select 
            value={settings.retrieval_method}
            onChange={(e) => handleChange('retrieval_method', e.target.value)}
            className="w-full text-sm border-gray-200 rounded-lg focus:ring-blue-500 focus:border-blue-500 bg-white"
          >
            <option value="random">Random baseline</option>
            <option value="caption_cosine">Caption cosine (Semantic)</option>
            <option value="caption_temporal">Caption + Temporal</option>
            <option value="siglip_direct">SigLIP direct (Semantic)</option>
            <option value="siglip_temporal">SigLIP + Temporal (Greedy)</option>
            <option value="siglip_temporal_hungarian">SigLIP + Temporal (Hungarian)</option>
            <option value="siglip_temporal_dp">SigLIP + Temporal (Viterbi/DP)</option>
            <option value="all">Compare all 6 arms</option>
          </select>
        </div>

        {/* Summary Style */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-gray-500">Summary style</label>
          <select 
            value={settings.style}
            onChange={(e) => handleChange('style', e.target.value)}
            className="w-full text-sm border-gray-200 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="informative">Informative</option>
            <option value="hook-driven">Hook-driven</option>
            <option value="educational">Educational</option>
          </select>
        </div>

        {/* Subtitles */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-gray-500">Subtitles</label>
          <select 
            value={settings.subtitles}
            onChange={(e) => handleChange('subtitles', e.target.value)}
            className="w-full text-sm border-gray-200 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="none">None</option>
            <option value="burn-in">Burn-in</option>
            <option value="srt">SRT only</option>
          </select>
        </div>

        {/* TTS Backend */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-gray-500">TTS backend</label>
          <select 
            value={settings.tts_backend}
            onChange={(e) => handleChange('tts_backend', e.target.value)}
            className="w-full text-sm border-gray-200 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="kokoro">Kokoro 1.0</option>
            <option value="dia">Dia 1.6B (Quality)</option>
            <option value="f5-tts">F5-TTS</option>
          </select>
        </div>

        {/* LLM Backend */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-gray-500">LLM model</label>
          <select 
            value={settings.llm_backend}
            onChange={(e) => handleChange('llm_backend', e.target.value)}
            className="w-full text-sm border-gray-200 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="groq">Llama 3.3</option>
            <option value="local">Qwen 2.5</option>
          </select>
        </div>
      </div>
    </div>
  );
}
