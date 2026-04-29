import json
from typing import List, Dict
from fastapi import WebSocket
import time

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast_to_job(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    # Connection might be closed
                    pass

manager = ConnectionManager()

class ProgressCallback:
    def __init__(self, job_id: str, websocket_manager: ConnectionManager, jobs_state: dict):
        self.job_id = job_id
        self.manager = websocket_manager
        self.jobs_state = jobs_state
        self.last_update_time = 0

    def update(self, phase: int, name: str, progress_pct: int, detail: str = "", vram_gb: float = 0):
        # Update in-memory state
        job = self.jobs_state.get(self.job_id)
        if not job:
            return

        # Check for cancellation
        if job.get("status") == "cancelling":
            job["status"] = "cancelled"
            from src.exceptions import JobCancelledError
            raise JobCancelledError(f"Job {self.job_id} cancelled by user")

        job["current_phase"] = phase
        job["phase_name"] = name
        job["progress_pct"] = progress_pct
        job["phase_details"] = detail
        
        # Calculate elapsed if not finished
        if job["status"] == "processing":
             job["elapsed_seconds"] = int(time.time() - job["start_time"])

        # Broadcast via WebSocket
        # To avoid flooding, we could throttle, but user asked for "at key points"
        # and "every 5 seconds" is handled by the loop in tasks.py usually.
        # But here we broadcast on every call.
        import asyncio
        
        message = {
            "phase": phase,
            "name": name,
            "progress_pct": progress_pct,
            "detail": detail,
            "vram_gb": vram_gb,
            "elapsed": job["elapsed_seconds"]
        }
        
        # Broadcast via the main loop safely from any thread
        try:
            # We assume the main loop is available via the manager's connections or globally
            # In FastAPI/Uvicorn, we can try to get the existing running loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Since broadcast_to_job is async, use run_coroutine_threadsafe if called from worker thread
                # but if we are already in the loop thread (rare for this callback), just create_task
                asyncio.run_coroutine_threadsafe(self.manager.broadcast_to_job(self.job_id, message), loop)
        except Exception:
            # Fallback for when loop is not available (e.g. initial job start)
            pass
