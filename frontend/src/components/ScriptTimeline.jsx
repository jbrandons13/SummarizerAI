import React from 'react';

export default function ScriptTimeline({ script, onSeek }) {
  return (
    <div className="card">
      <div className="text-sm font-semibold text-gray-900 border-b border-gray-100 pb-3 mb-4">Summary Script & Provenance</div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-gray-100 text-gray-500 font-medium">
              <th className="pb-2 w-10"> # </th>
              <th className="pb-2"> Narration Text </th>
              <th className="pb-2 w-24"> Source Timestamp </th>
              <th className="pb-2 w-24"> Similarity </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {script.map((line, idx) => (
              <tr 
                key={idx} 
                className="hover:bg-gray-50 cursor-pointer group transition-colors"
                onClick={() => onSeek(line.source_start)}
              >
                <td className="py-3 text-gray-400 font-mono">{idx + 1}</td>
                <td className="py-3 font-medium text-gray-900 leading-relaxed group-hover:text-blue-600 transition-colors">
                  {line.text}
                </td>
                <td className="py-3 text-gray-500 font-mono">
                  {line.source_start?.toFixed(1)}s - {line.source_end?.toFixed(1)}s
                </td>
                <td className="py-3">
                  <div className="flex items-center space-x-2">
                    <div className="w-12 bg-gray-100 rounded-full h-1 overflow-hidden">
                      <div 
                        className="bg-blue-400 h-full"
                        style={{ width: `${(line.similarity || 0) * 100}%` }}
                      />
                    </div>
                    <span className="text-gray-400 font-mono">{(line.similarity || 0).toFixed(2)}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
