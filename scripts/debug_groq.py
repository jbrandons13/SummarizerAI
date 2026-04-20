import groq
import os
from pathlib import Path

def load_env_manually():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env_manually()
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key found: {api_key[:10]}...")

try:
    client = groq.Groq(api_key=api_key)
    print("Client initialized successfully.")
    # Try a small call
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello"}],
        model="llama-3.3-70b-versatile",
    )
    print(f"Response: {chat_completion.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
