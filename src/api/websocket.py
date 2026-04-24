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
        
        # Since this is called from a thread, we need to run the broadcast in the event loop
        # We'll use the main event loop if possible, or just fire and forget.
        # FastAPI/Uvicorn run in an event loop.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.manager.broadcast_to_job(self.job_id, message))
        except Exception:
            pass
