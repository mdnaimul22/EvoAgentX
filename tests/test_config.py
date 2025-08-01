import sys
import os

# Add the project root to the Python path to allow for the `utils` import
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from openai import OpenAI, DefaultHttpxClient
from utils.config import client_rotator

print("--- Starting Configuration Test ---")

try:
    client_config = client_rotator.get_next_client_config()
    print("Successfully retrieved client configuration.")
    # Mask the API key for security
    masked_key = f"{'*' * (len(client_config.api_key) - 4)}{client_config.api_key[-4:]}" if len(client_config.api_key) > 4 else "****"
    print(f"  Model: {client_config.model}")
    print(f"  Base URL: {client_config.base_url}")
    print(f"  API Key: {masked_key}")
    print(f"  Proxy: {client_config.proxy}")
except Exception as e:
    print(f"[ERROR] Failed to get client configuration: {e}")
    sys.exit(1)


http_client = DefaultHttpxClient(proxy=client_config.proxy) if client_config.proxy else None
client = OpenAI(base_url=client_config.base_url, api_key=client_config.api_key, http_client=http_client, timeout=120.0)

system_prompt = "You are a helpful assistant."
user_prompt = "Hello! In one sentence, tell me who you are."

print("\nAttempting to create chat completion...")
try:
    response = client.chat.completions.create(
        model=client_config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=100,
        top_p=0.85,
        temperature=0.5,
    )
    print("\n[SUCCESS] Response received:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"\n[ERROR] An error occurred during the API call: {e}")

print("\n--- Test Finished ---")
