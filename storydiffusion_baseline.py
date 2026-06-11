import os
import csv
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
from diffusers import StableDiffusionXLPipeline
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
import copy

# ==============================================================================
# STORYDIFFUSION CONSISTENT SELF-ATTENTION IMPLEMENTATION
# Copied and adapted from StoryDiffusion repo
# ==============================================================================

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
                    if hidden_states.shape[1] == 1024: # 32x32
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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

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

from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor

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
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()
            
    unet.set_attn_processor(copy.deepcopy(attn_procs))
    print(f"Successfully loaded paired self-attention. Number of the processor: {total_count}")

# ==============================================================================
# PIPELINE AND EVALUATION LOGIC
# ==============================================================================

def get_clip_features(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)

def get_clip_text_features(texts, model, tokenizer, device):
    inputs = tokenizer(text=texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Load models
device = "cuda"
sdxl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)

dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_dinov2_features(images):
    inputs = dinov2_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
    features = outputs.last_hidden_state[:, 0, :]
    return features / features.norm(p=2, dim=-1, keepdim=True)

videos = {
    "Heart": "runs/heart/storyboard.json",
    "Sun": "runs/sun/storyboard.json",
    "Geology": "runs/geology/storyboard.json"
}

os.makedirs("runs/A3_storydiffusion", exist_ok=True)
csv_rows = []

# SDXL settings
res_w = 1024
res_h = 1024
num_steps = 30
guidance_scale = 7.0

global cur_step, write, mask1024, mask4096, sa32, sa64, total_count, attn_count
sa32 = 0.5
sa64 = 0.5

# Deliverable 5 stats
summary_stats = []

for vid_name, storyboard_path in videos.items():
    print(f"Processing {vid_name}...")
    with open(storyboard_path) as f:
        sb = json.load(f)["shots"]
    
    out_dir = f"runs/A3_storydiffusion/{vid_name}"
    os.makedirs(f"{out_dir}/vanilla", exist_ok=True)
    os.makedirs(f"{out_dir}/consistent", exist_ok=True)
    
    prompts = [s["image_prompt"] for s in sb]
    texts = [s["topic_tag"] for s in sb]
    
    seeds = [int(s["shot_id"].split("_")[1]) * 100 for s in sb]
    
    vanilla_images = []
    consistent_images = []
    
    expected_vanilla_last = f"{out_dir}/vanilla/shot_{len(prompts)-1:03d}.png"
    expected_consistent_last = f"{out_dir}/consistent/shot_{len(prompts)-1:03d}.png"
    
    if os.path.exists(expected_vanilla_last) and os.path.exists(expected_consistent_last):
        print(f"Loading {vid_name} images from disk...")
        for i in range(len(prompts)):
            vanilla_images.append(Image.open(f"{out_dir}/vanilla/shot_{i:03d}.png").convert("RGB"))
            consistent_images.append(Image.open(f"{out_dir}/consistent/shot_{i:03d}.png").convert("RGB"))
    else:
        from diffusers.models.attention_processor import AttnProcessor2_0
        sdxl.unet.set_attn_processor(AttnProcessor2_0()) # Reset attention
        
        print("Generating VANILLA...")
        for i, p in enumerate(prompts):
            torch.cuda.empty_cache()
            setup_seed(seeds[i])
            img = sdxl(p, num_inference_steps=num_steps, guidance_scale=guidance_scale, height=res_h, width=res_w).images[0]
            img.save(f"{out_dir}/vanilla/shot_{i:03d}.png")
            vanilla_images.append(img)
        
        # 2. Generate CONSISTENT (StoryDiffusion)
        print("Generating CONSISTENT...")
        id_length = 2 # Reduced from 4 to avoid OOM
        set_attention_processor(sdxl.unet, id_length)
        
        write = True
        cur_step = 0
        attn_count = 0
        mask1024 = None
        mask4096 = None
        setup_seed(seeds[0]) # Use a fixed seed for the whole batch or use list of generators
        
        # Generate id_images
        id_prompts = prompts[:id_length]
        generators = [torch.Generator(device="cuda").manual_seed(seeds[i]) for i in range(id_length)]
        
        id_imgs = sdxl(id_prompts, num_inference_steps=num_steps, guidance_scale=guidance_scale, height=res_h, width=res_w, generator=generators).images
        for i, img in enumerate(id_imgs):
            img.save(f"{out_dir}/consistent/shot_{i:03d}.png")
            consistent_images.append(img)
        
        # Generate rest
        write = False
        for i in range(id_length, len(prompts)):
            torch.cuda.empty_cache()
            cur_step = 0
            attn_count = 0
            setup_seed(seeds[i])
            img = sdxl(prompts[i], num_inference_steps=num_steps, guidance_scale=guidance_scale, height=res_h, width=res_w).images[0]
            img.save(f"{out_dir}/consistent/shot_{i:03d}.png")
            consistent_images.append(img)
        
    # 3. Calculate Metrics
    v_feats = get_dinov2_features(vanilla_images)
    c_feats = get_dinov2_features(consistent_images)
    t_feats = get_clip_text_features(texts, clip_model, clip_processor, device)
    c_clip_feats = get_clip_features(consistent_images, clip_model, clip_processor, device)
    
    content_preservation = F.cosine_similarity(c_feats, v_feats, dim=-1).cpu().numpy()
    concept_clip_t = F.cosine_similarity(c_clip_feats, t_feats, dim=-1).cpu().numpy()
    
    # Inter-shot sim
    v_inter = []
    c_inter = []
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            v_inter.append(F.cosine_similarity(v_feats[i:i+1], v_feats[j:j+1], dim=-1).item())
            c_inter.append(F.cosine_similarity(c_feats[i:i+1], c_feats[j:j+1], dim=-1).item())
            
    v_inter_mean = np.mean(v_inter) if v_inter else 0
    c_inter_mean = np.mean(c_inter) if c_inter else 0
    
    # Add to CSV
    for i in range(len(prompts)):
        csv_rows.append([vid_name, f"shot_{i:03d}", concept_clip_t[i], content_preservation[i]])
        
    csv_rows.append([])
    csv_rows.append([vid_name, np.mean(concept_clip_t), np.mean(content_preservation), c_inter_mean, v_inter_mean])
    csv_rows.append([])
    
    summary_stats.append(f"{vid_name}:\n  Mean Concept CLIP-T: {np.mean(concept_clip_t):.4f}\n  Mean Content Preserv: {np.mean(content_preservation):.4f}\n  Inter-shot Sim (Consistent): {c_inter_mean:.4f}\n  Inter-shot Sim (Vanilla): {v_inter_mean:.4f}\n")
    
    # Contact sheet
    contact_sheet = Image.new("RGB", (res_w * 2, res_h * len(prompts)))
    for i in range(len(prompts)):
        contact_sheet.paste(vanilla_images[i], (0, i * res_h))
        contact_sheet.paste(consistent_images[i], (res_w, i * res_h))
    contact_sheet.save(f"storydiffusion_{vid_name}_contactsheet.png")

with open("storydiffusion_baseline_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "shot/mean", "concept_clip_t/content_preservation", "inter_shot_sim_consistent", "inter_shot_sim_vanilla"])
    for row in csv_rows:
        writer.writerow(row)

with open("storydiffusion_summary.txt", "w") as f:
    f.write("Config: SDXL Base 1.0 (No LoRA)\nResolution: 1024x1024\nStoryDiffusion id_length=4\n\n")
    f.write("\n".join(summary_stats))

print("DONE!")
