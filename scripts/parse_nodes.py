import json

with open("/tmp/objects.json") as f:
    data = json.load(f)

nodes_to_check = ["UnetLoaderGGUF", "WanFirstLastFrameToVideo", "CLIPLoader", "VAELoader", "CLIPVisionLoader", "CLIPVisionEncode", "CLIPTextEncode", "KSampler", "VAEDecode", "SaveAnimatedWEBP", "LoadImage"]

for node in nodes_to_check:
    if node in data:
        print(f"--- {node} ---")
        for input_type, inputs in data[node].get("input", {}).items():
            print(f"  {input_type}:")
            for k, v in inputs.items():
                print(f"    {k}: {v}")
    else:
        print(f"--- {node} (MISSING) ---")
