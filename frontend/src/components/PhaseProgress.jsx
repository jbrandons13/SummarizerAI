import React from 'react';

export default function PhaseProgress({ phase, index, totalPhases }) {
  const isDone = phase.status === 'done';
  const isActive = phase.status === 'in_progress';
  const isWaiting = phase.status === 'waiting';

  return (
    <div className="flex items-start space-x-4">
      {/* Timeline Connector */}
      <div className="flex flex-col items-center flex-shrink-0 mt-1">
        <div className={`w-3 h-3 rounded-full border-2 ${
          isDone ? 'bg-green-500 border-green-500' : 
          isActive ? 'bg-blue-500 border-blue-500' : 'bg-white border-gray-300'
        }`} />
        {index < totalPhases - 1 && (
          <div className="w-0.5 h-16 bg-gray-200" />
        )}
      </div>

      {/* Content */}
      <div className="flex-grow">
        <div className="flex items-center justify-between mb-1">
          <span className={`text-sm font-semibold ${isWaiting ? 'text-gray-400' : 'text-gray-900'}`}>
            {phase.name}
          </span>
          <div className="flex items-center space-x-3">
             {phase.vram && <span className="text-[10px] text-gray-400 font-mono">Peak VRAM: {phase.vram}</span>}
             {phase.duration && <span className="text-[10px] text-gray-400 font-mono">{phase.duration}s</span>}
             <span className={`badge ${
               isDone ? 'badge-done' : isActive ? 'badge-active' : 'badge-waiting'
             }`}>
               {isDone ? 'Done' : isActive ? 'Processing' : 'Waiting'}
             </span>
          </div>
        </div>
        
        {/* Progress Bar Area */}
        <div className="mt-2 flex items-center space-x-3">
          <div className="progress-track overflow-hidden">
            <div 
              className={`progress-fill ${isDone ? 'bg-green-500' : 'bg-blue-500 animate-pulse'}`}
              style={{ width: `${phase.progress}%` }}
            />
          </div>
          <span className="text-[10px] font-mono text-gray-400 min-w-[30px]">{phase.progress}%</span>
        </div>
        
        {phase.detail && (
          <p className="mt-1.5 text-xs text-gray-500 leading-relaxed italic">
            {phase.detail}
          </p>
        )}
      </div>
    </div>
  );
}
