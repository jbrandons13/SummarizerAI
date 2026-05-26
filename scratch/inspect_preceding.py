import os
import diffusers

p_t2v = os.path.join(os.path.dirname(diffusers.__file__), 'pipelines/ltx/pipeline_ltx.py')
lines = open(p_t2v).readlines()
for i in range(700, 733):
    print(f"{i+1}: {lines[i]}", end="")
