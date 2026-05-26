import json

log_path = "/home/wins053/.gemini/antigravity/brain/e818488b-a4b9-4133-86db-0a25ff8037ed/.system_generated/logs/overview.txt"
output_script_path = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/scratch/extracted_ltx.py"

with open(log_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        if "run_remaining_ltx.py" in line and "write_to_file" in line:
            # Let's inspect the line
            try:
                data = json.loads(line)
                tool_calls = data.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("name") == "write_to_file":
                        args = tc.get("args", {})
                        target_file = args.get("TargetFile")
                        if "run_remaining_ltx.py" in target_file:
                            code_content = args.get("CodeContent")
                            # Write this out
                            with open(output_script_path, "w") as out:
                                out.write(code_content)
                            print(f"Success! Extracted from line {idx} to {output_script_path}")
            except Exception as e:
                print(f"Error parsing line {idx}: {e}")
