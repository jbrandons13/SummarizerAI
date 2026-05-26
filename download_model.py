from huggingface_hub import snapshot_download
import sys

repo_id = sys.argv[1]
print(f"Downloading {repo_id}...")
snapshot_download(repo_id=repo_id)
print("Done.")
