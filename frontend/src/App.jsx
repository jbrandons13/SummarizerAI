import React, { useState, useEffect } from 'react';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultsPage from './pages/ResultsPage';
import DashboardPage from './pages/DashboardPage';

function App() {
  const [currentPage, setCurrentPage] = useState('upload'); // upload | processing | results | dashboard
  const [currentJobId, setCurrentJobId] = useState(null);
  const [error, setError] = useState(null);

  const navigateTo = (page, jobId = null) => {
    if (jobId) setCurrentJobId(jobId);
    setCurrentPage(page);
    setError(null);
    window.scrollTo(0, 0);
  };

  const tabs = [
    { id: 'upload', label: 'Summarize' },
    { id: 'dashboard', label: 'Dashboard' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navigation Header */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 h-14 flex items-center justify-between">
          <div 
            className="text-lg font-bold text-blue-600 cursor-pointer"
            onClick={() => navigateTo('upload')}
          >
            SumarizerAI
          </div>
          <div className="flex space-x-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => navigateTo(tab.id)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  currentPage === tab.id || (currentPage === 'processing' && tab.id === 'upload') || (currentPage === 'results' && tab.id === 'upload')
                    ? 'text-blue-600 bg-blue-50'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow max-w-4xl mx-auto w-full px-4 py-8">
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-start space-x-3">
            <svg className="w-5 h-5 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="font-medium">Something went wrong</p>
              <p className="text-sm opacity-90">{error}</p>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {currentPage === 'upload' && (
            <UploadPage onStartJob={(jobId) => navigateTo('processing', jobId)} onError={setError} />
          )}

          {currentPage === 'processing' && (
            <ProcessingPage 
              jobId={currentJobId} 
              onComplete={() => navigateTo('results')} 
              onError={(msg) => { setError(msg); navigateTo('upload'); }}
            />
          )}

          {currentPage === 'results' && (
            <ResultsPage jobId={currentJobId} />
          )}

          {currentPage === 'dashboard' && (
            <DashboardPage />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 text-center text-gray-400 text-xs">
        &copy; 2026 SumarizerAI - Multimodal Video Summarization Pipeline
      </footer>
    </div>
  );
}

export default App;
