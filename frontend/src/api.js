const API_BASE = '/api';

export async function submitVideo(file, settings) {
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(settings).forEach(([k, v]) => {
    if (v !== undefined && v !== null) {
      formData.append(k, v);
    }
  });
  
  const res = await fetch(`${API_BASE}/summarize`, { 
    method: 'POST', 
    body: formData 
  });
  
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to submit video');
  }
  
  return res.json(); // { job_id }
}

export async function getStatus(jobId) {
  const res = await fetch(`${API_BASE}/status/${jobId}?t=${Date.now()}`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function getResult(jobId) {
  const res = await fetch(`${API_BASE}/result/${jobId}`);
  if (!res.ok) throw new Error('Failed to fetch result');
  return res.json();
}

export function connectProgress(jobId, onMessage, onError) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/progress/${jobId}`;
  
  const ws = new WebSocket(wsUrl);
  
  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      onMessage(data);
    } catch (err) {
      console.error('Failed to parse WS message', err);
    }
  };
  
  ws.onerror = (e) => {
    if (onError) onError(e);
  };
  
  return ws;
}

export async function getDashboard() {
  const res = await fetch(`${API_BASE}/eval/dashboard`);
  if (!res.ok) throw new Error('Failed to fetch dashboard');
  return res.json();
}

export async function exportEval() {
  const res = await fetch(`${API_BASE}/eval/export`);
  if (!res.ok) throw new Error('Failed to export evaluation');
  return res.blob();
}
export async function deleteJob(jobId) {
  const res = await fetch(`${API_BASE}/result/${jobId}`, {
    method: 'DELETE'
  });
  if (!res.ok) throw new Error('Failed to delete job');
  return res.json();
}

export async function cancelJob(jobId) {
  const res = await fetch(`${API_BASE}/cancel/${jobId}`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error('Failed to cancel job');
  return res.json();
}
