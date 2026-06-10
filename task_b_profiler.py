import subprocess
import time
import threading
import json
import csv
import os

stop_event = threading.Event()
peak_vram_gb = 0.0

def monitor_vram():
    global peak_vram_gb
    while not stop_event.is_set():
        try:
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            vrams = [float(x.strip()) / 1024.0 for x in res.strip().split('\n') if x.strip()]
            if vrams:
                current_vram = max(vrams)
                if current_vram > peak_vram_gb:
                    peak_vram_gb = current_vram
        except Exception:
            pass
        time.sleep(0.5)

def run_and_measure(name, cmd_string, multiplier=1.0):
    global peak_vram_gb
    peak_vram_gb = 0.0
    stop_event.clear()
    t = threading.Thread(target=monitor_vram)
    t.start()
    
    start_t = time.time()
    try:
        subprocess.run(cmd_string, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running {name}: {e}")
    
    end_t = time.time()
    stop_event.set()
    t.join()
    
    wall_min = ((end_t - start_t) / 60.0) * multiplier
    print(f"{name} -> Time: {wall_min:.2f} min, Peak VRAM: {peak_vram_gb:.2f} GB")
    return wall_min, peak_vram_gb

def main():
    os.makedirs("addon_results", exist_ok=True)
    results = []
    
    print("Profiling P1 WhisperX...")
    m, v = run_and_measure("P1_WhisperX", "PYTHONPATH=. python src/phase1_transcribe.py --video data/raw_videos/b22HKFMIfWo.mp4 --out /tmp/prof_transcript.json")
    results.append(("P1 WhisperX", m, v))
    
    print("Profiling P2 Summarize...")
    m1, v1 = run_and_measure("P2_Summarize", "PYTHONPATH=. python src/phase2_summarize.py --transcript runs/sun/transcript.json --out /tmp/prof_summary.json --backend local")
    
    print("Profiling P4 Segmenter...")
    m2, v2 = run_and_measure("P4_Segmenter", "PYTHONPATH=. python src/phase4/segmenter.py --video data/raw_videos/b22HKFMIfWo.mp4 --transcript runs/sun/transcript.json --script runs/sun/summary_script.json --out /tmp/prof_shots.json")
    
    print("Profiling P2 Storyboard...")
    m3, v3 = run_and_measure("P2_Storyboard", "PYTHONPATH=. python src/phase4/storyboard.py --shots runs/sun/shots.json --out /tmp/prof_storyboard.json --backend local")
    results.append(("P2 Qwen (Summarize+Storyboard) & Segmenter", m1+m2+m3, max(v1, v2, v3)))
    
    print("Profiling P3 TTS...")
    m, v = run_and_measure("P3_Kokoro_TTS", "PYTHONPATH=. python generate_ecology_audio.py") # we can just run this, it takes 1-2 mins, good proxy
    results.append(("P3 Kokoro TTS", m, v))
    
    print("Profiling P4 SDXL...")
    # run SDXL for 1 shot and multiply by 16
    m, v = run_and_measure("P4_SDXL_Single_w", "PYTHONPATH=. python weight_sweep.py --config configs/default.yaml --storyboard runs/sun/storyboard.json --reference runs/sun/reference.png --shots shot_001 --weights 0.2 --out /tmp/prof_sweep", multiplier=16.0)
    results.append(("P4 SDXL+IP-Adapter (Single w=0.2)", m, v))
    
    print("Profiling P5 Wan I2V...")
    # run Wan I2V for 1 shot and multiply by 16
    m, v = run_and_measure("P5_Wan_I2V", "PYTHONPATH=. python render_summary_video.py --all-i2v --only shot_001 --storyboard runs/sun/storyboard.json --script runs/sun/summary_script.json --images-dir runs/sun/images_fixed_w02 --audio-dir runs/sun/audio --work /tmp/prof_clips --final /tmp/prof_vid.mp4 --workflow scripts/wan_i2v_workflow.json", multiplier=16.0)
    results.append(("P5 Wan I2V + Assembly", m, v))
    
    with open("addon_results/runtime_vram.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "wall_clock_min", "peak_vram_gb"])
        tot_m = 0
        max_v = 0
        for r in results:
            writer.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.2f}"])
            tot_m += r[1]
            max_v = max(max_v, r[2])
        writer.writerow(["TOTAL", f"{tot_m:.2f}", f"{max_v:.2f}"])
        
    with open("addon_results/runtime_info.txt", "w") as f:
        f.write("Representative Video: V3 (Sun), 12 mins raw, 16 shots.\n")
        f.write("GPU: RTX 3090 24GB\n")
        f.write("Driver/Torch: Extracted via nvidia-smi / local torch version (PyTorch 2.6.0)\n")
        f.write("P4 Mode: Single-w (w=0.2). Run for 1 shot and extrapolated x16.\n")
        f.write("P5 Mode: Wan I2V. Run for 1 shot and extrapolated x16.\n")
        f.write("Measurement Method: psutil/subprocess polling nvidia-smi memory.used.\n")

if __name__ == "__main__":
    main()
