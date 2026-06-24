import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def query_vlm(images, text_prompt):
    messages = [{"role": "user", "content": []}]
    for img in images:
        messages[0]["content"].append({"type": "image", "image": img})
    messages[0]["content"].append({"type": "text", "text": text_prompt})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()

videos = {
    "V12_Eye": "a human eye",
    "V13_Hurricane": "a hurricane or a large swirling storm system",
    "V14_Reef": "a coral reef or underwater ecosystem"
}

for vid_prefix, target in videos.items():
    print(f"\nJudging {vid_prefix}...")
    paths = [
        f"gate_check/{vid_prefix}_reference.png",
        f"gate_check/{vid_prefix}_shot_1_w0.png",
        f"gate_check/{vid_prefix}_shot_2_w0.png",
        f"gate_check/{vid_prefix}_shot_3_w0.png"
    ]
    
    all_pass = True
    for p in paths:
        if not os.path.exists(p):
            print(f"  Missing {p}")
            all_pass = False
            continue
            
        img = Image.open(p).convert("RGB")
        prompt = f"Does this image clearly depict {target}? Answer Yes or No."
        ans = query_vlm([img], prompt)
        print(f"  {os.path.basename(p)}: {ans}")
        if "yes" not in ans.lower():
            all_pass = False
            
    if all_pass:
        print(f"  VERDICT: PASS")
    else:
        print(f"  VERDICT: FAIL")
