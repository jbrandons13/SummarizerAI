import os
import json
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderTiny

def main():
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
    
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").to("cuda", torch.float16)
    
    ref_path = "runs/geology/reference.png"
    ref_image = Image.open(ref_path).convert("RGB")
    
    shots = ["shot_006", "shot_011"]
    weights = [0.0, 0.4, 0.8]
    ks = [9, 15]
    
    os.makedirs("runs/step3_audit", exist_ok=True)
    
    # Monkey-patch scheduler to extract pred_x0
    original_step = pipe.scheduler.step
    def custom_step(model_output, timestep, sample, **kwargs):
        if "return_dict" in kwargs:
            del kwargs["return_dict"]
        res = original_step(model_output, timestep, sample, return_dict=True, **kwargs)
        pipe.scheduler._last_pred_x0 = res.pred_original_sample
        return (res.prev_sample,)
    pipe.scheduler.step = custom_step
    
    for shot_id in shots:
        shot_data = next(s for s in sb if s["shot_id"] == shot_id)
        prompt = "fca style, " + shot_data["image_prompt"]
        neg = "text, letters, words, numbers, captions, labels, infographic, diagram, panels, charts, table, watermark, signature, gibberish text"
        seed = seeds[f"geo_{shot_id.split('_')[1]}"]
        
        for w in weights:
            pipe.set_ip_adapter_scale(w)
            ip_img = _prep_reference(ref_image, mode="crop") if w > 0 else Image.new("RGB", (224, 224), "black")
            
            captured = {} # step -> {raw, pred_x0}
            
            def cb(p, i, t, cb_kwargs):
                step = i + 1 # so step 9 means i==8
                if step in ks:
                    captured[step] = {
                        "raw": cb_kwargs["latents"].clone(),
                        "pred_x0": p.scheduler._last_pred_x0.clone()
                    }
                return cb_kwargs
                
            gen = torch.Generator("cuda").manual_seed(seed)
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=1344, height=768,
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=gen,
                ip_adapter_image=ip_img,
                callback_on_step_end=cb
            ).images[0].resize((832, 480), Image.LANCZOS)
            
            # Now we have final image and captured latents at k=9, 15
            # We construct a mini contact sheet for this (shot, w)
            # Layout: [Final] [k=9 raw] [k=9 pred_x0] [k=15 raw] [k=15 pred_x0]
            
            row = Image.new("RGB", (832 * 5, 480))
            row.paste(out, (0, 0))
            
            for idx, k in enumerate(ks):
                c = captured[k]
                
                raw_scaled = (c["raw"] / pipe.vae.config.scaling_factor).to(torch.float16)
                with torch.no_grad():
                    raw_img = taesd.decode(raw_scaled).sample[0]
                raw_img = ((raw_img / 2 + 0.5).clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype("uint8")
                raw_img = Image.fromarray(raw_img).resize((832, 480), Image.LANCZOS)
                
                pred_scaled = (c["pred_x0"] / pipe.vae.config.scaling_factor).to(torch.float16)
                with torch.no_grad():
                    pred_img = taesd.decode(pred_scaled).sample[0]
                pred_img = ((pred_img / 2 + 0.5).clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype("uint8")
                pred_img = Image.fromarray(pred_img).resize((832, 480), Image.LANCZOS)
                
                row.paste(raw_img, (832 * (1 + idx*2), 0))
                row.paste(pred_img, (832 * (2 + idx*2), 0))
                
            row.save(f"runs/step3_audit/{shot_id}_w{w}.png")
            print(f"Saved runs/step3_audit/{shot_id}_w{w}.png")

if __name__ == "__main__":
    main()
