import os
import csv
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import gc
import copy
import sys
import matplotlib.pyplot as plt

# ==========================================================
# StoryDiffusion Consistent Self-Attention Processor
# ==========================================================
class SpatialAttnProcessor2_0(torch.nn.Module):
    def __init__(self, id_length=4, device="cuda", dtype=torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global cur_step, write, mask1024, mask4096, sa32, sa64, total_count, attn_count
        
        if write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length].detach().cpu(), hidden_states[self.id_length:].detach().cpu()]
        else:
            encoder_hidden_states = torch.cat(
                (self.id_bank[cur_step][0].to(self.device),
                 hidden_states[:1],
                 self.id_bank[cur_step][1].to(self.device),
                 hidden_states[1:])
            )
        
        if cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:
            random_number = random.random()
            rand_num = 0.3 if cur_step < 20 else 0.1
            
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == int((RES/32)**2):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == int((RES/32)**2):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,
                                                  :mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,
                                                  :mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(self.total_length, self.id_length, sa32, sa64, RES, RES, device=self.device, dtype=self.dtype)
            
        return hidden_states

    def __call1__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)

        batch_size, sequence_length, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(-1, (self.id_length + 1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call2__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, channel = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, sequence_length, channel).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

def cal_attn_mask_xl(total_length, id_length, sa32, sa64, height, width, device="cuda", dtype=torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024), device=device, dtype=dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096), device=device, dtype=dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length, 1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length, 1)
    for i in range(total_length):
        bool_matrix1024[i:i+1, id_length*nums_1024:] = False
        bool_matrix4096[i:i+1, id_length*nums_4096:] = False
        bool_matrix1024[i:i+1, i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1, i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1, nums_1024, 1).reshape(-1, total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1, nums_4096, 1).reshape(-1, total_length * nums_4096)
    return mask1024, mask4096

def set_attention_processor(unet, id_length):
    global total_count
    total_count = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
                total_count += 1
            else:    
                attn_procs[name] = AttnProcessor2_0()
        else:
            attn_procs[name] = AttnProcessor2_0()
            
    unet.set_attn_processor(copy.deepcopy(attn_procs))

VIDEOS_CONFIG = [
    ("Heart",   "runs/heart/storyboard.json",                  "all",    "fullrun_results/data/V4_Heart_adaptive_anchor.csv"),
    ("Sun",     "runs/sun/storyboard.json",                    "all",    "fullrun_results/data/V3_Sun_adaptive_anchor.csv"),
    ("Neuron",  "data/intermediate/V10/phase4/storyboard.json", "anchor", "fullrun_results/data/V10_Neuron_adaptive_anchor.csv"),
    ("Volcano", "data/intermediate/V11/phase4/storyboard.json", "anchor", "fullrun_results/data/V11_Volcano_adaptive_anchor.csv"),
]

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 30
GUIDANCE = 7.0
RES = 768 # Lower resolution to fit id_length=4 into 24GB VRAM
SWEEP_LENGTHS = [2, 3, 4]

def read_anchor(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    concept_text = rows[0][1].replace('"', '').strip()
    shot_ids = [r[0].replace('"', '').strip() for r in rows if len(r) > 0 and r[0].replace('"', '').strip().startswith("shot_")]
    return concept_text, shot_ids

def get_shots(sb_path, selection, anchor_csv):
    with open(sb_path, 'r') as f:
        data = json.load(f)
        shots = data["shots"] if "shots" in data else data
        
    concept_text, anchor_ids = read_anchor(anchor_csv)
    by_id = {s["shot_id"]: s["image_prompt"] for s in shots}
    
    ids = [s["shot_id"] for s in shots] if selection == "all" else anchor_ids
    ordered = [(sid, by_id[sid], int(sid.split("_")[1]) * 100) for sid in ids]
    return concept_text, ordered

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    global RES
    device = "cuda"
    out_dir = f"runs/id_length_sweep_res{RES}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading SDXL {SDXL_ID}...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(SDXL_ID, torch_dtype=torch.float16, use_safetensors=True).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    global cur_step, write, mask1024, mask4096, sa32, sa64, total_count, attn_count
    sa32, sa64 = 0.5, 0.5
    
    vram_peaks = {}
    
    for vid_name, sb_path, selection, anchor_csv in VIDEOS_CONFIG:
        print(f"--- Processing Video: {vid_name} ---")
        concept_text, ordered = get_shots(sb_path, selection, anchor_csv)
        n_shots = len(ordered)
        
        v_dir = os.path.join(out_dir, vid_name, "vanilla")
        os.makedirs(v_dir, exist_ok=True)
        
        # 1. VANILLA (no dependence on id_length)
        if not os.path.exists(os.path.join(v_dir, f"shot_{n_shots-1:03d}.png")):
            print(f"[{vid_name}] Generating VANILLA set...")
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            for i, (sid, prompt, seed) in enumerate(ordered):
                torch.cuda.empty_cache()
                set_seed(seed)
                img = pipeline(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES).images[0]
                img.save(os.path.join(v_dir, f"shot_{i:03d}.png"))
                
        # 2. CONSISTENT sweep
        for idl in SWEEP_LENGTHS:
            c_dir = os.path.join(out_dir, vid_name, f"consistent_id{idl}")
            os.makedirs(c_dir, exist_ok=True)
            if os.path.exists(os.path.join(c_dir, f"shot_{n_shots-1:03d}.png")):
                continue
                
            print(f"[{vid_name}] Generating CONSISTENT set (id_length={idl})...")
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            gc.collect()
            torch.cuda.empty_cache()
            
            try:
                set_attention_processor(pipeline.unet, idl)
                write = True
                cur_step = 0
                attn_count = 0
                mask1024, mask4096 = None, None
                
                id_prompts = [ordered[i][1] for i in range(idl)]
                generators = [torch.Generator(device=device).manual_seed(ordered[i][2]) for i in range(idl)]
                set_seed(ordered[0][2])
                id_imgs = pipeline(id_prompts, num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES, generator=generators).images
                for i, img in enumerate(id_imgs):
                    img.save(os.path.join(c_dir, f"shot_{i:03d}.png"))
                    
                write = False
                for i in range(idl, n_shots):
                    torch.cuda.empty_cache()
                    cur_step = 0
                    attn_count = 0
                    set_seed(ordered[i][2])
                    img = pipeline(ordered[i][1], num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES).images[0]
                    img.save(os.path.join(c_dir, f"shot_{i:03d}.png"))
                    
                pipeline.unet.set_attn_processor(AttnProcessor2_0())
                vram_peaks[idl] = max(vram_peaks.get(idl, 0), torch.cuda.max_memory_allocated() / 1e9)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM AT ID_LENGTH={idl}! Exiting so resolution can be lowered.")
                    sys.exit(1)
                raise e

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------
    print("Loading DINOv2 for measurement...")
    dinov2 = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    
    def get_dino(imgs):
        inputs = dinov2_proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = dinov2(**inputs).last_hidden_state[:, 0, :]
        return f / f.norm(p=2, dim=-1, keepdim=True)
        
    results = []
    
    for vid_name, sb_path, selection, anchor_csv in VIDEOS_CONFIG:
        concept_text, ordered = get_shots(sb_path, selection, anchor_csv)
        n_shots = len(ordered)
        v_dir = os.path.join(out_dir, vid_name, "vanilla")
        v_imgs = [Image.open(os.path.join(v_dir, f"shot_{i:03d}.png")).convert("RGB") for i in range(n_shots)]
        v_feats = get_dino(v_imgs)
        
        v_inter = []
        for i in range(n_shots):
            for j in range(i+1, n_shots):
                v_inter.append(F.cosine_similarity(v_feats[i:i+1], v_feats[j:j+1]).item())
        mean_v_inter = np.mean(v_inter) if v_inter else 0
        
        for idl in SWEEP_LENGTHS:
            c_dir = os.path.join(out_dir, vid_name, f"consistent_id{idl}")
            c_imgs = [Image.open(os.path.join(c_dir, f"shot_{i:03d}.png")).convert("RGB") for i in range(n_shots)]
            c_feats = get_dino(c_imgs)
            
            preservation = F.cosine_similarity(c_feats, v_feats, dim=-1).cpu().numpy()
            mean_pres = np.mean(preservation)
            
            c_inter = []
            for i in range(n_shots):
                for j in range(i+1, n_shots):
                    c_inter.append(F.cosine_similarity(c_feats[i:i+1], c_feats[j:j+1]).item())
            mean_c_inter = np.mean(c_inter) if c_inter else 0
            
            homog = mean_c_inter - mean_v_inter
            results.append({
                "video": vid_name,
                "id_length": idl,
                "resolution": RES,
                "mean_content_preservation": mean_pres,
                "inter_shot_consistent": mean_c_inter,
                "inter_shot_vanilla": mean_v_inter,
                "homogenization": homog
            })

    # Save CSV
    csv_file = "id_length_sweep_metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "id_length", "resolution", "mean_content_preservation", "inter_shot_consistent", "inter_shot_vanilla", "homogenization"])
        writer.writeheader()
        writer.writerows(results)
        
    with open("id_length_sweep_vram.txt", "w") as f:
        f.write(f"Sweep Resolution: {RES}x{RES}\n")
        for k, v in vram_peaks.items():
            f.write(f"ID Length {k} Peak VRAM: {v:.2f} GB\n")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for vid_name in [v[0] for v in VIDEOS_CONFIG]:
        v_res = [r for r in results if r["video"] == vid_name]
        v_res.sort(key=lambda x: x["id_length"])
        xs = [r["id_length"] for r in v_res]
        
        y_homog = [r["homogenization"] for r in v_res]
        y_content = [r["mean_content_preservation"] for r in v_res]
        
        ax1.plot(xs, y_homog, marker='o', label=vid_name)
        ax2.plot(xs, y_content, marker='o', label=vid_name)
        
    ax1.set_title("Homogenization vs id_length")
    ax1.set_xlabel("id_length")
    ax1.set_ylabel("Homogenization (InterSim C - InterSim V)")
    ax1.set_xticks(SWEEP_LENGTHS)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Content Preservation vs id_length")
    ax2.set_xlabel("id_length")
    ax2.set_ylabel("Content Preservation (cosine with Vanilla)")
    ax2.set_xticks(SWEEP_LENGTHS)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("id_length_sweep_plot.png", dpi=150)
    print("DONE SWEEP!")

if __name__ == "__main__":
    main()
