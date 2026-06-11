import os
import json
import torch
import yaml
import time
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderTiny
from pipeline.facet.scoring_wrap import ScoringWrap

def main():
    os.makedirs("runs/G1_Alt1_v2", exist_ok=True)
    
    with open("configs/facet.yaml", "r") as f:
        facet_config = yaml.safe_load(f)
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("pipeline/facet/seeds.json", "r") as f:
        seeds = json.load(f)
        
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]
        
    from src.phase4.image_gen import load_pipeline, _prep_reference
    
    pipe, _ = load_pipeline(config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Monkey-patch scheduler
    original_step = pipe.scheduler.step
    def custom_step(model_output, timestep, sample, **kwargs):
        if "return_dict" in kwargs:
            del kwargs["return_dict"]
        res = original_step(model_output, timestep, sample, return_dict=True, **kwargs)
        pipe.scheduler._last_pred_x0 = res.pred_original_sample
        return (res.prev_sample,)
    pipe.scheduler.step = custom_step

    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").to("cuda", torch.float16)
    scorer = ScoringWrap()
    
    ref_path = "runs/geology/reference.png"
    ref_image = Image.open(ref_path).convert("RGB")
    
    weights = [0.0] + facet_config["W_grid"] # [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    shots = [f"geo_{i:03d}" for i in range(1, 15)]
    ks = [9, 12]
    
    global_text = "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones"
    
    matrices = {
        k: {s: {w: {} for w in weights} for s in shots} for k in ks
    }
    
    total_calls = 0
    t0 = time.time()
    
    for k in ks:
        print(f"--- Running k={k} ---")
        for shot_id in shots:
            shot_num = shot_id.split("_")[1]
            shot_data = next(s for s in sb if s["shot_id"] == f"shot_{shot_num}")
            prompt = "fca style, " + shot_data["image_prompt"]
            neg = "text, letters, words, numbers, captions, labels, infographic, diagram, panels, charts, table, watermark, signature, gibberish text"
            seed = seeds[shot_id]
            
            w0_preview_emb = None
            
            for w in weights:
                pipe.set_ip_adapter_scale(w)
                ip_img = _prep_reference(ref_image, mode="crop") if w > 0 else Image.new("RGB", (224, 224), "black")
                
                captured_pred_x0 = None
                
                def cb(p, i, t, cb_kwargs):
                    nonlocal captured_pred_x0
                    step = i + 1
                    if step == k:
                        captured_pred_x0 = p.scheduler._last_pred_x0.clone()
                        raise StopIteration()
                    return cb_kwargs
                    
                gen = torch.Generator("cuda").manual_seed(seed)
                try:
                    pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        width=1344, height=768,
                        num_inference_steps=30,
                        guidance_scale=7.0,
                        generator=gen,
                        ip_adapter_image=ip_img,
                        callback_on_step_end=cb
                    )
                except StopIteration:
                    pass
                    
                total_calls += k
                
                # Decode pred_x0
                pred_scaled = (captured_pred_x0 / pipe.vae.config.scaling_factor).to(torch.float16)
                with torch.no_grad():
                    pred_img = taesd.decode(pred_scaled).sample[0]
                pred_img = ((pred_img / 2 + 0.5).clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype("uint8")
                img = Image.fromarray(pred_img).resize((832, 480), Image.LANCZOS)
                
                img_path = f"runs/G1_Alt1_v2/{shot_id}_w{w}_k{k}.png"
                img.save(img_path)
                
                # Score
                mc = scorer.get_clip_concept(img_path, global_text)
                emb = scorer.embed_dino(img_path)
                
                if w == 0.0:
                    w0_preview_emb = emb
                    c_hat = 1.0
                else:
                    c_hat = float((emb * w0_preview_emb).sum().item())
                    
                matrices[k][shot_id][w] = {
                    "preview_mc": mc,
                    "c_hat": c_hat
                }
                
                if time.time() - t0 > 600: # 10 minute heartbeat
                    print(f"Heartbeat: {total_calls} UNet calls so far...")
                    t0 = time.time()
                    
        # Save checkpoints
        with open(f"runs/G1_Alt1_v2/matrices_k{k}.json", "w") as f:
            json.dump(matrices[k], f, indent=2)

if __name__ == "__main__":
    main()
