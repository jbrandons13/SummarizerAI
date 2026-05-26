import os
import diffusers

p_t2v = os.path.join(os.path.dirname(diffusers.__file__), 'pipelines/ltx/pipeline_ltx.py')
p_i2v = os.path.join(os.path.dirname(diffusers.__file__), 'pipelines/ltx/pipeline_ltx_image2video.py')

def print_call_site(p, name):
    if os.path.exists(p):
        print(f"=== {name} ({p}) ===")
        lines = open(p).readlines()
        for i, line in enumerate(lines):
            if "retrieve_timesteps" in line:
                # Print 5 lines before and after
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                for j in range(start, end):
                    print(f"{j+1}: {lines[j]}", end="")
    else:
        print(f"{name} path does not exist: {p}")

print_call_site(p_t2v, "T2V Pipeline")
print_call_site(p_i2v, "I2V Pipeline")
