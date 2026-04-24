import uvicorn
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

if __name__ == "__main__":
    # Create necessary directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/intermediate").mkdir(parents=True, exist_ok=True)
    Path("data/output").mkdir(parents=True, exist_ok=True)
    
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
