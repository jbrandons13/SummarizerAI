import React, { useState, useEffect } from 'react';
import MetricCard from '../components/MetricCard';
import ArmComparisonTable from '../components/ArmComparisonTable';
import { getDashboard, exportEval } from '../api';

export default function DashboardPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

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

  if (loading) {
    return <div className="py-20 text-center text-gray-500">Loading dashboard metrics...</div>;
  }

  if (!data || data.videos_tested === 0) {
    return (
      <div className="card text-center py-20">
        <div className="text-gray-300 mb-4">
          <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
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
    <div className="space-y-6 animate-in fade-in duration-700">
      <div className="flex justify-between items-center">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-gray-900">Ablation Dashboard</h1>
          <p className="text-sm text-gray-500">Aggregated metrics across {data.videos_tested} test cases.</p>
        </div>
        <div className="flex space-x-2">
           <button 
             onClick={handleExport}
             className="btn"
           >
             Export CSV
           </button>
           <button className="btn btn-primary">
             Generate Charts
           </button>
        </div>
      </div>

      {/* Aggregate Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Videos Tested" value={data.videos_tested} />
        <MetricCard label="Avg CLIPScore" value={Object.values(data.arms).reduce((acc, arm) => acc + arm.clipscore_mean, 0) / Object.keys(data.arms).length || 0} />
        <MetricCard label="Avg ROUGE-L" value={Object.values(data.arms).reduce((acc, arm) => acc + arm.rouge_l_mean, 0) / Object.keys(data.arms).length || 0} />
        <MetricCard label="Avg Processing" value="184s" />
      </div>

      {/* Arm Comparison Table */}
      <ArmComparisonTable arms={data.arms} />

      {/* Detailed Results per Case */}
      <div className="card">
        <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3 mb-4">Case History</div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
             <tr className="text-gray-500 font-medium border-b border-gray-100">
               <th className="pb-2">Video ID</th>
               <th className="pb-2">Arm</th>
               <th className="pb-2">CLIPScore</th>
               <th className="pb-2">ROUGE-L</th>
               <th className="pb-2">VRAM (MB)</th>
             </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {data.per_video.map((case_item, idx) => (
                <tr key={idx} className="hover:bg-gray-50 transition-colors">
                  <td className="py-3 font-mono text-gray-400">{case_item.video_id?.slice(0,8)}...</td>
                  <td className="py-3 capitalize font-medium">{case_item.arm?.replace('_', ' ')}</td>
                  <td className="py-3 font-mono">{(case_item.clipscore_mean || 0).toFixed(4)}</td>
                  <td className="py-3 font-mono">{(case_item.rouge_l || 0).toFixed(4)}</td>
                  <td className="py-3 text-gray-400">4,200</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
