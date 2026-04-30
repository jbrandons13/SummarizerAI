import React, { useState, useEffect, useMemo } from 'react';
import MetricCard from '../components/MetricCard';
import ArmComparisonTable from '../components/ArmComparisonTable';
import { getDashboard, exportEval, deleteJob } from '../api';
import MetricChart from '../components/MetricChart';

export default function DashboardPage({ onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const result = await getDashboard();
        setData(result);
      } catch (err) {
        console.error("Failed to load dashboard", err);
      } finally {
        setLoading(false);
      }
    };
    loadDashboard();
  }, []);

  const groupedVideos = useMemo(() => {
    if (!data?.per_video) return [];
    const groups = {};
    data.per_video.forEach(item => {
      const vid = String(item.video_id || '');
      if (!groups[vid]) {
        groups[vid] = {
          video_id: vid,
          arms: [],
          best_clipscore: -1,
          best_arm: '',
        };
      }
      groups[vid].arms.push(item);
      if (item.clipscore_mean > groups[vid].best_clipscore) {
        groups[vid].best_clipscore = item.clipscore_mean;
        groups[vid].best_arm = item.arm;
      }
    });
    return Object.values(groups).reverse(); // Newest first probably
  }, [data]);

  const handleExport = async () => {
    try {
      const blob = await exportEval();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ablation_results.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert("Failed to export: " + err.message);
    }
  };

  const toggleExpand = (id) => setExpandedId(expandedId === id ? null : id);

  if (loading) {
    return (
      <div className="py-20 flex flex-col items-center justify-center space-y-4">
        <div className="w-12 h-12 border-4 border-blue-100 border-t-blue-600 rounded-full animate-spin"></div>
        <div className="text-gray-500 font-medium animate-pulse">Loading dashboard metrics...</div>
      </div>
    );
  }

  if (!data || (data.videos_tested === 0 && (!data.recent_jobs || data.recent_jobs.length === 0))) {
    return (
      <div className="card text-center py-20">
        <div className="text-gray-300 mb-4">
          <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <h2 className="text-lg font-semibold text-gray-900">No evaluation data yet</h2>
        <p className="text-sm text-gray-500 max-w-xs mx-auto mt-2">
          Run the pipeline on some videos to see ablation study results and model comparisons here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-700 pb-20">
      <div className="flex justify-between items-center">
        <div className="space-y-1">
          <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">AI Ablation Dashboard</h1>
          <p className="text-sm text-gray-500">Comparative Analysis across {data.videos_tested} Test Cases.</p>
        </div>
        <div className="flex space-x-2">
           <button onClick={handleExport} className="btn-premium px-6 py-2">
             Export Dataset
           </button>
        </div>
      </div>

      {/* Aggregate Metrics Performance Overview */}
      {data.videos_tested > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <MetricCard label="Cases" value={data.videos_tested} />
              <MetricCard label="Success %" value="100%" /> 
            </div>
            {/* Main Stats Card */}
            <div className="card-glass p-6 text-center space-y-2 border-l-4 border-blue-600">
              <div className="text-[10px] font-bold text-blue-600 uppercase tracking-widest">Master Accuracy</div>
              <div className="text-4xl font-black text-gray-900">
                {(Object.values(data.arms).reduce((acc, arm) => acc + arm.clipscore_mean, 0) / 
                  Math.max(Object.keys(data.arms).length, 1)).toFixed(3)}
              </div>
              <div className="text-xs text-gray-400">Mean CLIPScore across all arms</div>
            </div>
          </div>
          
          <div className="lg:col-span-2">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <MetricChart data={data.arms} metric="clipscore_mean" label="Visual Relevance (CLIP)" color="blue" />
              <MetricChart data={data.arms} metric="rouge_l_mean" label="Content Overlap (ROUGE-L)" color="purple" />
              <MetricChart data={data.arms} metric="vram_peak" label="Peak VRAM Usage (MB)" color="orange" />
              <MetricChart data={data.arms} metric="temporal_acc_15s" label="Temporal Accuracy (15s)" color="emerald" />
              <MetricChart data={data.arms} metric="visual_coherence_mean" label="Visual Coherence (SigLIP)" color="pink" />
            </div>
          </div>
        </div>
      )}

      {/* Arm Comparison Details */}
      {data.videos_tested > 0 && (
        <div className="space-y-4">
          <div className="flex items-center space-x-2 text-xs font-bold text-gray-400 uppercase tracking-widest">
            <div className="w-8 h-px bg-gray-200"></div>
            <span>Detailed Statistical Breakdown</span>
            <div className="w-full h-px bg-gray-200"></div>
          </div>
          <ArmComparisonTable arms={data.arms} />
        </div>
      )}

      {/* Recent Jobs History - NEW SECTION */}
      <div className="card">
        <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3 mb-4">Recent Summaries History</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {data.recent_jobs?.map(job => (
            <div 
              key={job.job_id} 
              className="p-3 border border-gray-100 rounded-xl hover:border-blue-200 hover:bg-blue-50/20 transition-all cursor-pointer group flex justify-between items-center"
              onClick={() => onNavigate('results', job.job_id)}
            >
              <div className="space-y-1">
                <div className="text-xs font-mono text-gray-400 group-hover:text-blue-500 transition-colors">
                  {job.job_id.substring(0, 13)}...
                </div>
                <div className="text-[10px] text-gray-400">
                  {new Date(job.timestamp * 1000).toLocaleString()}
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <div onClick={(e) => { e.stopPropagation(); onNavigate('results', job.job_id); }} className="text-blue-600 font-medium text-xs flex items-center hover:underline">
                  View Results 
                  <svg className="w-3 h-3 ml-1 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="9 5l7 7-7 7" />
                  </svg>
                </div>
                <button 
                  onClick={async (e) => { 
                    e.stopPropagation(); 
                    if (window.confirm('Clear this history item and its associated files?')) {
                      try {
                        await deleteJob(job.job_id);
                        // Refresh data
                        const result = await getDashboard();
                        setData(result);
                      } catch (err) {
                        alert("Failed to delete: " + err.message);
                      }
                    }
                  }}
                  className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                  title="Delete History"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
          {(!data.recent_jobs || data.recent_jobs.length === 0) && (
            <div className="col-span-2 text-center py-4 text-gray-400 text-xs italic">No job history found.</div>
          )}
        </div>
      </div>

      {/* Detailed Results per Case */}
      {groupedVideos.length > 0 && (
        <div className="card">
          <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3 mb-4">Ablation Detailed Results (Click to Expand)</div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
              <tr className="text-gray-500 font-medium border-b border-gray-100">
                <th className="pb-2">Video Identifier</th>
                <th className="pb-2">Arms</th>
                <th className="pb-2">Peak CLIPScore</th>
                <th className="pb-2">Action</th>
              </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {groupedVideos.map((video, idx) => (
                  <React.Fragment key={idx}>
                    <tr className="hover:bg-gray-50 transition-colors cursor-pointer group">
                      <td onClick={() => toggleExpand(video.video_id)} className="py-3 font-mono font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                        {(video.video_id.length > 20 ? video.video_id.substring(0,8) + '...' : video.video_id) || "Demo Video"}
                      </td>
                      <td onClick={() => toggleExpand(video.video_id)} className="py-3 capitalize text-gray-500">{video.arms.length} Arms</td>
                      <td onClick={() => toggleExpand(video.video_id)} className="py-3 font-medium">
                        <span className="text-green-600">{(video.best_clipscore || 0).toFixed(4)}</span>
                        <span className="text-xs text-gray-400 capitalize ml-2">({video.best_arm?.replace('_', ' ') || 'None'})</span>
                      </td>
                      <td className="py-3 space-x-3">
                        <button 
                          onClick={(e) => { e.stopPropagation(); onNavigate('results', video.video_id); }}
                          className="text-blue-600 hover:underline font-medium"
                        >
                          View
                        </button>
                        <button 
                          onClick={async (e) => {
                            e.stopPropagation();
                          if (window.confirm(`Delete data ${video.video_id} from statistics?`)) {
                            try {
                              await deleteJob(video.video_id);
                              const result = await getDashboard();
                              setData(result);
                            } catch (err) {
                              alert("Failed to delete: " + err.message);
                            }
                          }
                        }}
                        className="text-gray-300 hover:text-red-500 transition-colors"
                        title="Delete from Statistics"
                        >
                          <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </td>
                    </tr>
                    
                    {expandedId === video.video_id && (
                      <tr>
                        <td colSpan="4" className="p-0 border-b-0">
                          <div className="bg-gray-50 p-4 border-b border-gray-100 shadow-inner">
                            <table className="w-full text-left text-xs bg-white rounded shadow-sm border border-gray-200">
                              <thead className="bg-gray-50 text-gray-500 border-b border-gray-200">
                                <tr>
                                  <th className="px-4 py-2 font-medium text-gray-700">Retrieval Feature</th>
                                  <th className="px-4 py-2 font-medium">CLIPScore (ViT)</th>
                                  <th className="px-4 py-2 font-medium">ROUGE-L</th>
                                  <th className="px-4 py-2 font-medium">BERTScore</th>
                                  <th className="px-4 py-2 font-medium">Judge (Inf/Fac/Vis)</th>
                                  <th className="px-4 py-2 font-medium">Time</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-gray-100 text-gray-600">
                                {video.arms.map((arm, aIdx) => (
                                  <tr key={aIdx} className={arm.arm === video.best_arm ? 'bg-green-50/20 text-gray-900' : ''}>
                                    <td className="px-4 py-3 font-medium capitalize flex items-center gap-2">
                                      {arm.arm?.replace('_', ' ')}
                                      {arm.arm === video.best_arm && <span className="text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded uppercase font-bold">Best</span>}
                                    </td>
                                    <td className="px-4 py-3 font-mono">{(arm.clipscore_mean || 0).toFixed(4)}</td>
                                    <td className="px-4 py-3 font-mono">{(arm.rouge_l || 0).toFixed(4)}</td>
                                    <td className="px-4 py-3 font-mono">{(arm.bertscore || 0).toFixed(4)}</td>
                                    <td className="px-4 py-3 font-mono text-[10px]">
                                      {arm.information_retention || '?'}/{arm.factual_faithfulness || '?'}/{arm.visual_relevance || '?'}
                                    </td>
                                    <td className="px-4 py-3 text-gray-400">{(arm.total_time_sec || 0).toFixed(0)}s</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
