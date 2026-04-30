import React from 'react';

export default function ArmComparisonTable({ arms }) {
  if (!arms || Object.keys(arms).length === 0) return null;

  const metrics = [
    { key: 'clipscore_mean', label: 'CLIPScore', higherBetter: true },
    { key: 'rouge_l_mean', label: 'ROUGE-L (Info Retention)', higherBetter: true },
    { key: 'bertscore_mean', label: 'BERTScore (Semantic)', higherBetter: true },
    { key: 'processing_time', label: 'Processing Time', higherBetter: false },
    { key: 'vram_peak', label: 'Peak VRAM (MB)', higherBetter: false },
    { key: 'temporal_acc_15s', label: 'Temporal Acc (15s)', higherBetter: true },
    { key: 'visual_coherence_mean', label: 'Visual Coherence', higherBetter: true },
  ];

  const armList = Object.keys(arms);

  const getWinner = (metricKey, higherBetter) => {
    let bestArm = null;
    let bestValue = higherBetter ? -Infinity : Infinity;

    armList.forEach(arm => {
      const val = arms[arm][metricKey];
      if (val === undefined) return;
      if (higherBetter) {
        if (val > bestValue) {
          bestValue = val;
          bestArm = arm;
        }
      } else {
        if (val < bestValue) {
          bestValue = val;
          bestArm = arm;
        }
      }
    });
    return bestArm;
  };

  return (
    <div className="card">
      <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3 mb-4">Ablation Metrics Comparison</div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-gray-100 text-gray-500 font-medium">
              <th className="pb-2"> Metric </th>
              {armList.map(arm => (
                <th key={arm} className="pb-2 capitalize"> {arm.replace('_', ' ')} </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {metrics.map(metric => {
              const winner = getWinner(metric.key, metric.higherBetter);
              return (
                <tr key={metric.key}>
                  <td className="py-3 text-gray-500 font-medium">{metric.label}</td>
                  {armList.map(arm => {
                    const value = arms[arm][metric.key];
                    const isWinner = arm === winner;
                    return (
                      <td key={arm} className="py-3">
                        <div className="flex items-center space-x-2">
                          <span className={`${isWinner ? 'text-emerald-600 font-bold' : 'text-gray-900'}`}>
                            {value !== undefined ? (typeof value === 'number' ? value.toFixed(4) : value) : '-'}
                          </span>
                          {isWinner && <span className="badge badge-best scale-75 origin-left">WINNER</span>}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
