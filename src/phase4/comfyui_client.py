import json
import time
import os
import urllib.request
import urllib.error
from pathlib import Path
import subprocess
import signal
import uuid
import websocket

class ComfyUIClient:
    def __init__(self, host="127.0.0.1", port=8188, comfyui_path="~/comfyui/ComfyUI"):
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self._server_process = None
        self.comfyui_path = Path(os.path.expanduser(comfyui_path))
    
    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"
    
    def is_running(self, timeout=2) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False

    def start_server(self, wait_for_ready=120) -> None:
        if self.is_running():
            return
        
        log_file = open("/tmp/comfy_api.log", "w")
        self._server_process = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", "comfyui", "python", "main.py", "--listen", self.host, "--port", str(self.port)],
            cwd=str(self.comfyui_path),
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        start_time = time.time()
        while time.time() - start_time < wait_for_ready:
            if self.is_running():
                return
            time.sleep(2)
        
        raise RuntimeError("ComfyUI server failed to start within the timeout.")

    def kill_server(self) -> None:
        if self._server_process:
            self._server_process.send_signal(signal.SIGTERM)
            self._server_process.wait()
            self._server_process = None

    def queue_workflow(self, workflow: dict) -> str:
        p = {"prompt": workflow, "client_id": self.client_id}
        req = urllib.request.Request(f"{self.base_url}/prompt", data=json.dumps(p).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read())
                return res['prompt_id']
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTPError: {e.read().decode('utf-8')}")

    def wait_for_completion(self, prompt_id: str, timeout=2400) -> dict:
        ws = websocket.WebSocket()
        try:
            ws.connect(f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}", timeout=10)
        except Exception as e:
            return {"status": "error", "details": f"WebSocket connection failed: {e}"}
        
        start_time = time.time()
        ws.settimeout(timeout)
        try:
            while time.time() - start_time < timeout:
                try:
                    out = ws.recv()
                except websocket.WebSocketTimeoutException:
                    break
                
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            ws.close()
                            return {"status": "success", "details": "Execution completed"}
                    elif message['type'] == 'execution_error':
                        data = message['data']
                        err_msg = str(data.get('exception_message', '')).lower()
                        exception_type = str(data.get('exception_type', '')).lower()
                        traceback = str(data.get('traceback', '')).lower()
                        is_oom = "out of memory" in err_msg or "oom" in err_msg or "cuda error" in err_msg or "out of memory" in traceback
                        
                        ws.close()
                        status = "oom" if is_oom else "error"
                        return {"status": status, "details": data}
            ws.close()
            return {"status": "error", "details": "Timeout waiting for completion"}
        except Exception as e:
            ws.close()
            return {"status": "error", "details": f"WebSocket error: {e}"}

    def get_output(self, prompt_id: str, output_node_id: str = "14") -> dict:
        req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}")
        try:
            with urllib.request.urlopen(req) as response:
                history = json.loads(response.read())
                if prompt_id in history:
                    outputs = history[prompt_id].get('outputs', {})
                    if output_node_id in outputs and 'images' in outputs[output_node_id]:
                        return outputs[output_node_id]['images'][0]
                return {}
        except Exception:
            return {}

    def download_output(self, filename: str, subfolder: str, dest_path: Path) -> None:
        url = f"{self.base_url}/view?filename={filename}&subfolder={subfolder}"
        urllib.request.urlretrieve(url, dest_path)

    def free_vram(self) -> None:
        req = urllib.request.Request(f"{self.base_url}/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(req)
        except Exception:
            pass

    def system_stats(self) -> dict:
        req = urllib.request.Request(f"{self.base_url}/system_stats")
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read())
        except Exception:
            return {}
