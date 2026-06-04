import json
import time
import os
import shutil
import urllib.request
import urllib.error
from pathlib import Path
import subprocess
import signal
import uuid
import websocket

# Config
VIDEO_ID = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
BASE_PATH = Path(f"data/intermediate/{VIDEO_ID}/phase4")
FRAME_PATH = BASE_PATH / "concept_anchor_canonical_w02/images/shot_016.png"
COMFY_DIR = Path(os.path.expanduser("~/comfyui/ComfyUI"))
COMFY_INPUT_DIR = COMFY_DIR / "input"
COMFY_OUTPUT_DIR = COMFY_DIR / "output"
OUTPUT_PATH = BASE_PATH / "_smoke_test/shot_016_w02_i2v.webp"
CLIENT_ID = str(uuid.uuid4())

def is_comfyui_running(port=8188):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/system_stats")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

def check_preflight():
    print("Checking models...")
    models = [
        COMFY_DIR / "models/unet/wan2.1-i2v-14b-720p-Q5_K_M.gguf",
        COMFY_DIR / "models/text_encoders/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        COMFY_DIR / "models/vae/split_files/vae/wan_2.1_vae.safetensors",
        COMFY_DIR / "models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors"
    ]
    for m in models:
        if not m.exists():
            print(f"❌ Missing model: {m}")
            return False
    
    if not FRAME_PATH.exists():
        print(f"❌ Missing input image {FRAME_PATH}")
        return False
    print("✅ Preflight check passed")
    return True

def submit_prompt(workflow):
    p = {"prompt": workflow, "client_id": CLIENT_ID}
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps(p).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.read().decode('utf-8')}")
        raise

def get_history(prompt_id):
    req = urllib.request.Request(f"http://127.0.0.1:8188/history/{prompt_id}")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception:
        return {}

def get_system_stats():
    req = urllib.request.Request("http://127.0.0.1:8188/system_stats")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception:
        return {}

def track_websocket(prompt_id, start_time):
    ws = websocket.WebSocket()
    ws.connect(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}")
    
    print("WebSocket connected. Waiting for execution...")
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    print("\n✅ Execution complete!")
                    break # Execution is done
                else:
                    print(f"\n[+{int(time.time() - start_time)}s] Executing Node {data['node']}")
            elif message['type'] == 'execution_error':
                data = message['data']
                print(f"\n❌ EXECUTION ERROR at node {data['node_id']}: {data['exception_type']}")
                print(data['exception_message'])
                print(data['traceback'])
                ws.close()
                return False
            elif message['type'] == 'progress':
                data = message['data']
                print(f"\rProgress: {data['value']}/{data['max']}", end="")
    ws.close()
    return True

def main():
    if not check_preflight():
        return
    
    comfy_proc = None
    if not is_comfyui_running():
        print("Starting ComfyUI background process...")
        log_file = open("/tmp/comfy_api.log", "w")
        comfy_proc = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", "comfyui", "python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
            cwd=str(COMFY_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        for _ in range(120):
            if is_comfyui_running():
                print("ComfyUI started successfully")
                break
            time.sleep(1)
        else:
            print("❌ Timeout waiting for ComfyUI to start")
            return
            
    try:
        print("Copying inputs to ComfyUI/input...")
        COMFY_INPUT_DIR.mkdir(exist_ok=True)
        shutil.copy(FRAME_PATH, COMFY_INPUT_DIR / "smoke_i2v_frame.png")
        
        # Load workflow
        with open("scripts/wan_i2v_workflow.json") as f:
            workflow = json.load(f)
            
        # Get prompt
        with open(BASE_PATH / "storyboard.json") as f:
            storyboard = json.load(f)
            shot_016 = next(s for s in storyboard["shots"] if s["shot_id"] == "shot_016")
            prompt_text = shot_016["visual_description"]
            
        workflow["5"]["inputs"]["image"] = "smoke_i2v_frame.png"
        workflow["9"]["inputs"]["text"] = prompt_text
        workflow["11"]["inputs"]["length"] = 53  # 3.4 seconds at 16fps = ~54. We use 53
        workflow["14"]["inputs"]["filename_prefix"] = "smoke_test_shot_016_w02_i2v"
        
        print(f"Prompt: {prompt_text}")
        print("Submitting workflow...")
        start_time = time.time()
        res = submit_prompt(workflow)
        prompt_id = res['prompt_id']
        print(f"Prompt ID: {prompt_id}")
        
        success = track_websocket(prompt_id, start_time)
        gen_time = time.time() - start_time
        print(f"Total time elapsed: {gen_time:.1f}s")
        
        # Check VRAM Peak via system_stats
        stats = get_system_stats()
        vram_peak = 0
        if stats and "devices" in stats:
            for dev in stats["devices"]:
                if dev.get("type") == "cuda":
                    vram_peak = dev.get("vram_total", 0) - dev.get("vram_free", 0)
        print(f"Current VRAM Usage: {vram_peak / 1e9:.2f} GB")
        
        if not success:
            print("❌ Workflow failed.")
            return

        # Fetch history for output
        history = get_history(prompt_id)
        if prompt_id in history:
            result = history[prompt_id]
            outputs = result.get('outputs', {})
            # Node 14 is SaveAnimatedWEBP
            if "14" in outputs and "images" in outputs["14"]:
                out_filename = outputs["14"]["images"][0]["filename"]
                out_path = COMFY_OUTPUT_DIR / out_filename
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(out_path, OUTPUT_PATH)
                print(f"✅ Success! Saved to {OUTPUT_PATH}")
                print(f"Size: {OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")
                
                # Copy it to artifacts directory
                try:
                    art_dir = Path("/home/wins053/.gemini/antigravity/brain/e8bcb447-cc25-4828-8f6a-b05b1a04592e/artifacts/")
                    art_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(out_path, art_dir / "shot_016_w02_i2v.webp")
                    print(f"Copied to artifacts dir.")
                except Exception as e:
                    print(f"Failed to copy to artifacts: {e}")
            else:
                print("❌ Output not found in history!")
                print(json.dumps(outputs, indent=2))
        else:
            print("❌ History not found for prompt_id")

    finally:
        # We leave comfyUI running so we can kill it manually later if needed, but it's safe to kill it
        if comfy_proc:
            print("Killing ComfyUI process...")
            comfy_proc.send_signal(signal.SIGTERM)
            comfy_proc.wait()

if __name__ == "__main__":
    main()
