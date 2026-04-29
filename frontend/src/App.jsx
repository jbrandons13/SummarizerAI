import React, { useState, useEffect } from 'react';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultsPage from './pages/ResultsPage';
import DashboardPage from './pages/DashboardPage';

function App() {
  const [currentPage, setCurrentPage] = useState('upload'); // upload | processing | results | dashboard
  const [currentJobId, setCurrentJobId] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const navigateTo = (page, jobId = null) => {
    // Safety guard for processing
    if (currentPage === 'processing' && page !== 'processing' && page !== 'results') {
      const confirm = window.confirm("A job is currently in progress. Navigating away will STOP and CANCEL the process. Continue?");
      if (!confirm) return;
      
      // Auto-cancel the job if user leaves
      if (currentJobId) {
        import('./api').then(api => {
          api.cancelJob(currentJobId).catch(err => console.error("Auto-cancel failed:", err));
        });
      }
      setIsProcessing(false);
    }

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
          <div className="flex items-center space-x-4">
            <div 
              className="text-lg font-bold text-blue-600 cursor-pointer flex items-center"
              onClick={() => navigateTo('upload')}
            >
              SumarizerAI
            </div>
            
            {isProcessing && currentPage !== 'processing' && (
              <button 
                onClick={() => navigateTo('processing')}
                className="flex items-center space-x-2 px-3 py-1 bg-blue-50 text-blue-600 text-xs font-semibold rounded-full border border-blue-200 animate-pulse hover:bg-blue-100 transition-colors"
                title="Click to view progress"
              >
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                <span>Active Job in Progress</span>
              </button>
            )}
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
              <p className="text-sm opacity-90 mb-3">{error}</p>
              <button 
                onClick={() => { setError(null); navigateTo('upload'); }}
                className="px-4 py-1.5 bg-red-100 hover:bg-red-200 text-red-800 text-xs font-bold rounded-lg transition-colors border border-red-300 active:scale-[0.98]"
              >
                Back to Home
              </button>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {currentPage === 'upload' && (
            <UploadPage 
              onStartJob={(jobId) => { setIsProcessing(true); navigateTo('processing', jobId); }} 
              onError={setError} 
            />
          )}

          {currentPage === 'processing' && (
            <ProcessingPage 
              jobId={currentJobId} 
              onComplete={() => { setIsProcessing(false); navigateTo('results'); }} 
              onError={(msg) => { setIsProcessing(false); setError(msg); navigateTo('upload'); }}
            />
          )}

          {currentPage === 'results' && (
            <ResultsPage jobId={currentJobId} />
          )}

          {currentPage === 'dashboard' && (
            <DashboardPage onNavigate={navigateTo} />
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
