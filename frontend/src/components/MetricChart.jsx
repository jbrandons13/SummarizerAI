import React from 'react';

export default function MetricChart({ data, metric, label, color = 'blue' }) {
  if (!data || Object.keys(data).length === 0) return null;

  const arms = Object.keys(data);
  const values = arms.map(arm => data[arm][metric] || 0);
  const maxValue = Math.max(...values, 0.1);

  const colors = {
    blue: 'from-blue-600 to-indigo-500',
    purple: 'from-purple-600 to-fuchsia-500',
    green: 'from-emerald-500 to-teal-400',
    orange: 'from-orange-500 to-amber-400',
    emerald: 'from-emerald-600 to-green-400',
    pink: 'from-pink-600 to-rose-400',
    teal: 'from-teal-600 to-cyan-400'
  };

  return (
    <div className="card-glass p-5 space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wider">{label}</h3>
        <div className="text-[10px] text-gray-400 font-mono">Comparison by Method</div>
      </div>
      
      <div className="space-y-4 pt-2">
        {arms.map((arm, i) => {
          const val = data[arm][metric] || 0;
          const pct = (val / maxValue) * 100;
          
          return (
            <div key={arm} className="space-y-1">
              <div className="flex justify-between text-[10px] font-medium text-gray-500 uppercase">
                <span>{arm.replace('_', ' ')}</span>
                <span className="font-mono text-gray-900">{val.toFixed(3)}</span>
              </div>
              <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                <div 
                  className={`h-full bg-gradient-to-r ${colors[color]} rounded-full transition-all duration-1000 ease-out`}
                  style={{ width: `${pct}%`, transitionDelay: `${i * 100}ms` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
