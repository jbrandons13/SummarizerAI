import React from 'react';
import VideoPlayer from './VideoPlayer';

export default function ComparisonView({ method, outputs, onSeekAll }) {
  const isComparison = method === 'all';
  
  // Find highest CLIPScore for 'best' badge
  let bestArm = null;
  if (isComparison && outputs) {
    let maxScore = -1;
    Object.entries(outputs).forEach(([arm, data]) => {
      if (data.clipscore > maxScore) {
        maxScore = data.clipscore;
        bestArm = arm;
      }
    });
  }

  const handleRef = (arm) => (ref) => {
    // Optional: Syncing logic if needed
  };

  return (
    <div className="card">
      <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3 mb-4">
        {isComparison ? 'Ablation Study: Retrieval Method Comparison' : 'Summary Results'}
      </div>
      
      <div className={`grid gap-4 ${isComparison ? 'grid-cols-1 md:grid-cols-3' : 'grid-cols-1'}`}>
        {isComparison ? (
          <>
            {['random', 'caption_cosine', 'siglip_direct'].map(arm => (
              <div key={arm} className="space-y-3">
                <VideoPlayer 
                  url={outputs[arm]?.video_url} 
                  label={arm.replace('_', ' ')} 
                  best={arm === bestArm}
                />
                {outputs[arm] && (
                  <div className="text-center">
                    <span className="text-[10px] text-gray-400 font-mono">CLIPScore: </span>
                    <span className="text-sm font-semibold text-blue-600">{(outputs[arm].clipscore || 0).toFixed(4)}</span>
                  </div>
                )}
              </div>
            ))}
          </>
        ) : (
          <VideoPlayer 
            url={outputs[method]?.video_url} 
            label={method.replace('_', ' ')} 
          />
        )}
      </div>
    </div>
  );
}
