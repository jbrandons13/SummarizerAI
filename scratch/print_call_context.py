import os
import diffusers

p_t2v = os.path.join(os.path.dirname(diffusers.__file__), 'pipelines/ltx/pipeline_ltx.py')
p_i2v = os.path.join(os.path.dirname(diffusers.__file__), 'pipelines/ltx/pipeline_ltx_image2video.py')

def print_context(p, start_line, end_line, name):
    if os.path.exists(p):
        print(f"=== {name} lines {start_line}-{end_line} ===")
        lines = open(p).readlines()
        for i in range(start_line - 1, min(len(lines), end_line)):
            print(f"{i+1}: {lines[i]}", end="")

print_context(p_t2v, 730, 745, "T2V Pipeline")
print_context(p_i2v, 800, 815, "I2V Pipeline")
