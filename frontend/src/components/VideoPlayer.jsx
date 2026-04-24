import React, { useRef, useEffect } from 'react';

export default function VideoPlayer({ url, label, best, onRef }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (onRef) onRef(videoRef.current);
  }, [onRef]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-tighter">{label}</span>
        {best && <span className="badge badge-best">Best Retrieval</span>}
      </div>
      <div className="relative rounded-lg overflow-hidden border border-gray-200 bg-black aspect-video flex items-center justify-center">
        {!url ? (
          <div className="text-gray-600 text-xs italic">Loading video...</div>
        ) : (
          <video 
            ref={videoRef}
            src={url} 
            controls 
            className="w-full h-full object-contain"
            preload="metadata"
          />
        )}
      </div>
    </div>
  );
}
