import React from 'react';

export default function MetricCard({ label, value, subvalue, icon }) {
  return (
    <div className="metric-card border border-gray-100 items-center justify-center flex flex-col">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1 font-medium">{label}</div>
      <div className="text-2xl font-semibold text-gray-900">{value}</div>
      {subvalue && <div className="text-xs text-gray-400 mt-1">{subvalue}</div>}
    </div>
  );
}
