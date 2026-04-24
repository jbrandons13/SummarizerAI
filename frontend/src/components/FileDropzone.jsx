import React, { useCallback, useState } from 'react';

export default function FileDropzone({ onFileSelect, selectedFile }) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div 
      className={`card border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center min-h-[200px] ${
        isDragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => document.getElementById('fileInput').click()}
    >
      <input 
        id="fileInput"
        type="file" 
        className="hidden" 
        accept=".mp4,.mkv,.webm,.mov"
        onChange={handleInputChange}
      />
      
      {!selectedFile ? (
        <div className="text-center">
          <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4 text-gray-400">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <p className="text-sm font-medium text-gray-900">Click to upload or drag and drop</p>
          <p className="text-xs text-gray-500 mt-1">MP4, MKV, WebM or MOV (max. 500MB)</p>
        </div>
      ) : (
        <div className="flex items-center space-x-4">
          <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <div className="text-left">
            <p className="text-sm font-medium text-gray-900 truncate max-w-[200px]">{selectedFile.name}</p>
            <p className="text-xs text-gray-400">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
          <button 
            type="button"
            className="text-gray-400 hover:text-red-500"
            onClick={(e) => {
              e.stopPropagation();
              onFileSelect(null);
            }}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}
