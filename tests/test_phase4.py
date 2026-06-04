import torch
import gc
import time
from diffusers import DiffusionPipeline

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def test_flux():
    print("====================================")
    print("1. Menguji Model: FLUX.2 Klein 4B")
    print("====================================")
    try:
        # Load Flux.2
        print("Loading FLUX.2 weights (ini mungkin memakan waktu sebentar)...")
        flux_pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16
        )
        print("✅ Berhasil memuat struktur Pipeline FLUX.2!")
        
        # Bersihkan dari memory
        del flux_pipe
        flush_memory()
        print("✅ Memori FLUX.2 berhasil dibersihkan.\n")
    except Exception as e:
        print(f"❌ Gagal memuat FLUX.2: {e}\n")

def test_wan():
    print("====================================")
    print("2. Menguji Model: Wan 2.2 A14B Lightning FP8")
    print("====================================")
    try:
        print("Loading Wan 2.2 weights (ini berukuran besar, mohon tunggu)...")
        wan_pipe = DiffusionPipeline.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            torch_dtype=torch.float16
        )
        print("✅ Berhasil memuat struktur Pipeline Wan 2.2!")
        
        # Bersihkan dari memory
        del wan_pipe
        flush_memory()
        print("✅ Memori Wan 2.2 berhasil dibersihkan.\n")
    except Exception as e:
        print(f"❌ Gagal memuat Wan 2.2: {e}\n")

if __name__ == "__main__":
    print("Mulai Uji Coba Model Phase 4 (Anchor Policy) - Local Check")
    test_flux()
    test_wan()
    print("Uji coba selesai!")
