import ast
import json
import os

input_file = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/extracted_ltx_script.py"
output_file = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/scratch/run_ltx.py"

with open(input_file, "r") as f:
    content = f.read().strip()

# Attempt to decode using ast.literal_eval (for python string representation)
# or json.loads (if it's a JSON string)
try:
    code = ast.literal_eval(content)
except Exception as e:
    print(f"ast.literal_eval failed: {e}. Trying json.loads...")
    try:
        code = json.loads(content)
    except Exception as e2:
        print(f"json.loads failed: {e2}. Fallback to replacing literal newlines...")
        # Fallback raw replace
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        code = content.replace('\\n', '\n').replace('\\"', '"')

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    f.write(code)

print("Decoded script written successfully to", output_file)
