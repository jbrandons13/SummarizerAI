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
                    if hidden_states.shape[1] == 1024:
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == 1024:
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
            mask1024, mask4096 = cal_attn_mask_xl(self.total_length, self.id_length, sa32, sa64, 1024, 1024, device=self.device, dtype=self.dtype)
            
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

# ==========================================================
# Baseline Experiment Script
# ==========================================================

VIDEOS_CONFIG = [
    {"name": "Heart", "storyboard": "runs/heart/storyboard.json", "anchor": "fullrun_results/data/V4_Heart_adaptive_anchor.csv"},
    {"name": "Eye", "storyboard": "data/intermediate/V12/phase4/storyboard.json", "anchor": "fullrun_results/data/V12_Eye_adaptive_anchor.csv"},
    {"name": "Sun", "storyboard": "runs/sun/storyboard.json", "anchor": "fullrun_results/data/V3_Sun_adaptive_anchor.csv"}
]

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
ID_LENGTH = 2
STEPS = 30
GUIDANCE = 7.0
RES = 1024

def read_anchor(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    concept_text = rows[0][1].replace('"', '').strip()
    daca_concept, daca_content = 0, 0
    for r in rows:
        if len(r) > 0 and 'adaptive' in r[0]:
            daca_concept = float(r[1])
            daca_content = float(r[2])
    return concept_text, daca_concept, daca_content

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    device = "cuda"
    out_dir = "runs/external_baseline_3videos"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading SDXL {SDXL_ID}...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(SDXL_ID, torch_dtype=torch.float16, use_safetensors=True).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    global cur_step, write, mask1024, mask4096, sa32, sa64, total_count, attn_count
    sa32, sa64 = 0.5, 0.5
    
    for vid in VIDEOS_CONFIG:
        vid_name = vid["name"]
        print(f"--- Processing Video: {vid_name} ---")
        
        with open(vid["storyboard"]) as f:
            data = json.load(f)
            shots = data["shots"] if "shots" in data else data
            
        prompts = [s["image_prompt"] for s in shots]
        seeds = [int(s["shot_id"].split("_")[1]) * 100 for s in shots]
        n_shots = len(prompts)
        
        v_dir = os.path.join(out_dir, vid_name, "vanilla")
        c_dir = os.path.join(out_dir, vid_name, "consistent")
        os.makedirs(v_dir, exist_ok=True)
        os.makedirs(c_dir, exist_ok=True)
        
        # VANILLA
        print(f"[{vid_name}] Generating VANILLA set...")
        pipeline.unet.set_attn_processor(AttnProcessor2_0())
        for i in range(n_shots):
            fp = os.path.join(v_dir, f"shot_{i:03d}.png")
            if not os.path.exists(fp):
                torch.cuda.empty_cache()
                set_seed(seeds[i])
                img = pipeline(prompts[i], num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES).images[0]
                img.save(fp)
                
        # CONSISTENT (StoryDiffusion)
        print(f"[{vid_name}] Generating CONSISTENT set...")
        last_c = os.path.join(c_dir, f"shot_{n_shots-1:03d}.png")
        if not os.path.exists(last_c):
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            gc.collect()
            torch.cuda.empty_cache()
            
            set_attention_processor(pipeline.unet, ID_LENGTH)
            write = True
            cur_step = 0
            attn_count = 0
            mask1024, mask4096 = None, None
            
            generators = [torch.Generator(device=device).manual_seed(seeds[i]) for i in range(ID_LENGTH)]
            set_seed(seeds[0])
            id_imgs = pipeline(prompts[:ID_LENGTH], num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES, generator=generators).images
            for i, img in enumerate(id_imgs):
                img.save(os.path.join(c_dir, f"shot_{i:03d}.png"))
                
            write = False
            for i in range(ID_LENGTH, n_shots):
                torch.cuda.empty_cache()
                cur_step = 0
                attn_count = 0
                set_seed(seeds[i])
                img = pipeline(prompts[i], num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=RES, width=RES).images[0]
                img.save(os.path.join(c_dir, f"shot_{i:03d}.png"))
                
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------
    print("Loading models for measurement...")
    dinov2 = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def get_dino(imgs):
        inputs = dinov2_proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = dinov2(**inputs).last_hidden_state[:, 0, :]
        return f / f.norm(p=2, dim=-1, keepdim=True)
        
    def get_clip_img(imgs):
        inputs = clip_proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = clip.get_image_features(**inputs)
        return f / f.norm(p=2, dim=-1, keepdim=True)
        
    def get_clip_txt(txt):
        inputs = clip_proc(text=[txt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            f = clip.get_text_features(**inputs)
        return f / f.norm(p=2, dim=-1, keepdim=True)
        
    metrics_per_shot = []
    metrics_agg = []
    fair_coords = []
    summary_lines = []
    
    for vid in VIDEOS_CONFIG:
        vid_name = vid["name"]
        concept_text, daca_c, daca_p = read_anchor(vid["anchor"])
        
        with open(vid["storyboard"]) as f:
            data = json.load(f)
            shots = data["shots"] if "shots" in data else data
        n_shots = len(shots)
        
        v_dir = os.path.join(out_dir, vid_name, "vanilla")
        c_dir = os.path.join(out_dir, vid_name, "consistent")
        
        v_imgs = [Image.open(os.path.join(v_dir, f"shot_{i:03d}.png")).convert("RGB") for i in range(n_shots)]
        c_imgs = [Image.open(os.path.join(c_dir, f"shot_{i:03d}.png")).convert("RGB") for i in range(n_shots)]
        
        v_feats = get_dino(v_imgs)
        c_feats = get_dino(c_imgs)
        txt_feat = get_clip_txt(concept_text)
        c_clip_feats = get_clip_img(c_imgs)
        
        preservation = F.cosine_similarity(c_feats, v_feats, dim=-1).cpu().numpy()
        concept = F.cosine_similarity(c_clip_feats, txt_feat.expand_as(c_clip_feats), dim=-1).cpu().numpy()
        
        v_inter, c_inter = [], []
        for i in range(n_shots):
            for j in range(i+1, n_shots):
                v_inter.append(F.cosine_similarity(v_feats[i:i+1], v_feats[j:j+1]).item())
                c_inter.append(F.cosine_similarity(c_feats[i:i+1], c_feats[j:j+1]).item())
                
        mean_v_inter = np.mean(v_inter) if v_inter else 0
        mean_c_inter = np.mean(c_inter) if c_inter else 0
        mean_pres = np.mean(preservation)
        mean_conc = np.mean(concept)
        
        for i in range(n_shots):
            metrics_per_shot.append([vid_name, f"shot_{i:03d}", f"{concept[i]:.4f}", f"{preservation[i]:.4f}"])
            
        metrics_agg.append([vid_name, f"{mean_conc:.4f}", f"{mean_pres:.4f}", f"{mean_c_inter:.4f}", f"{mean_v_inter:.4f}"])
        fair_coords.append([vid_name, f"{mean_conc:.4f}", f"{mean_pres:.4f}", f"{daca_c:.4f}", f"{daca_p:.4f}"])
        
        summary_lines.append(f"[{vid_name}] Concept={mean_conc:.4f} Content={mean_pres:.4f} InterSim(C)={mean_c_inter:.4f} InterSim(V)={mean_v_inter:.4f}")
        
        # Contact sheet
        sheet = Image.new("RGB", (RES*2, RES*n_shots), "white")
        for i in range(n_shots):
            sheet.paste(v_imgs[i], (0, i*RES))
            sheet.paste(c_imgs[i], (RES, i*RES))
        sheet = sheet.resize((1024, 512*n_shots))
        sheet.save(f"external_{vid_name}_contactsheet.png")
        
    with open("external_baseline_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "shot", "concept_clip_t_consistent", "content_preservation_consistent"])
        writer.writerows(metrics_per_shot)
        writer.writerow([])
        writer.writerow(["video", "mean_concept_clip_t", "mean_content_preservation", "inter_shot_sim_consistent", "inter_shot_sim_vanilla"])
        writer.writerows(metrics_agg)
        writer.writerow([])
        writer.writerow(["# Fair-plane Coordinates"])
        writer.writerow(["video", "baseline_concept", "baseline_content", "daca_concept", "daca_content"])
        writer.writerows(fair_coords)
        
    with open("external_baseline_summary.txt", "w") as f:
        f.write("StoryDiffusion Baseline (id_length=2)\n")
        f.write("\n".join(summary_lines))
        f.write(f"\nPeak VRAM: {vram_peak:.2f} GB\n")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, coord in zip(axes, fair_coords):
            vid, bc, bp, dc, dp = coord
            bc, bp, dc, dp = float(bc), float(bp), float(dc), float(dp)
            ax.scatter([dc], [dp], marker="*", s=200, label="DACA")
            ax.scatter([bc], [bp], marker="o", s=100, label="Baseline")
            ax.set_title(vid)
            ax.legend()
        fig.savefig("external_fairplane.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
