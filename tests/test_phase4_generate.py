import torch
import gc
import os
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def generate_random_scene():
    # Prompt simpel untuk testing
    prompt = "A cinematic shot of a cute golden retriever dog walking in a cyberpunk neon city, highly detailed, 4k"
    
    # Path output
    os.makedirs("results/phase4_tests", exist_ok=True)
    image_path = "results/phase4_tests/test_shot_01.png"
    video_path = "results/phase4_tests/test_shot_01.mp4"
    
    print("====================================")
    print("=== TEST PIPELINE PHASE 4 LOKAL ===")
    print("====================================\n")
    print(f"Prompt Target: '{prompt}'\n")

    # ---------------------------------------------------------
    # 1. TEXT-TO-IMAGE (FLUX.2)
    # ---------------------------------------------------------
    print("[1/2] 🚀 Memuat FLUX.2 Klein 4B ke VRAM...")
    try:
        flux_pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16
        )
        # Pindahkan ke GPU
        flux_pipe.to("cuda")
        
        print("🎨 Mulai merender gambar (menggunakan 15 langkah agar cepat)...")
        # Perbaikan: Menggunakan kwargs prompt=prompt
        image = flux_pipe(
            prompt=prompt,
            num_inference_steps=15,
            guidance_scale=3.5,
            height=480, 
            width=848
        ).images[0]
        
        image.save(image_path)
        print(f"✅ Sukses! Gambar *Anchor* T2I disimpan di: {image_path}")
        
        # Unload model dari VRAM agar tidak OOM saat meload Wan 14B
        del flux_pipe
        flush_memory()
        print("🧹 Memori VRAM (Flux) berhasil dibersihkan.\n")
        
    except Exception as e:
        print(f"❌ Error di proses T2I: {e}")
        return

    # ---------------------------------------------------------
    # 2. IMAGE-TO-VIDEO (WAN 2.1 14B)
    # ---------------------------------------------------------
    print("[2/2] 🚀 Memuat Wan 2.1 14B ke VRAM...")
    try:
        wan_pipe = DiffusionPipeline.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            torch_dtype=torch.float16
        )
        wan_pipe.to("cuda")

        print("🎬 Mulai menganimasikan gambar dari Flux menjadi Video...")
        print("   (Ini memakan waktu sekitar 3-8 menit di RTX 3090, mohon tunggu...)")
        
        output = wan_pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=20, # Step rendah untuk test awal
            height=480,
            width=848
        ).frames[0]

        export_to_video(output, video_path, fps=16)
        print(f"\n✅ Sukses! Video I2V berhasil disimpan di: {video_path}")
        
        # Unload
        del wan_pipe
        flush_memory()
        print("🧹 Memori VRAM (Wan) berhasil dibersihkan.")
        
    except Exception as e:
        print(f"❌ Error di proses I2V: {e}")

if __name__ == "__main__":
    generate_random_scene()
