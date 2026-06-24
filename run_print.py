import subprocess
import sys
res = subprocess.run(["conda", "run", "-n", "sumarizer", "python", "print_vae.py"], capture_output=True, text=True)
print(res.stdout)
print(res.stderr, file=sys.stderr)
