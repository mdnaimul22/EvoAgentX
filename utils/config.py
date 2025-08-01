"""
Usages:
from openai import OpenAI, DefaultHttpxClient
from utils.config import client_rotator

client_config = client_rotator.get_next_client_config()
http_client = DefaultHttpxClient(proxy=client_config.proxy) if client_config.proxy else None
client = OpenAI(base_url=client_config.base_url, api_key=client_config.api_key, http_client=http_client, timeout=120.0)

response = client.chat.completions.create(
    model=client_config.model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    max_tokens=8000,
    top_p=0.85,
    temperature=0.5,
)
"""

import os
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ClientConfig:
    base_url: str
    api_key: str
    model: str
    proxy: Optional[str] = None

class ClientRotator:
    def __init__(self, client_configs: List[ClientConfig]):
        if not client_configs:
            raise ValueError("Client configurations list cannot be empty.")
        self.client_configs = client_configs
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_client_config(self) -> ClientConfig:
        with self.lock:
            config = self.client_configs[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.client_configs)
            return config

# Load configurations from environment variables
# Example:
# CLIENT_CONFIGS_0_BASE_URL="http://localhost:8000/v1"
# CLIENT_CONFIGS_0_API_KEY="sk-abcde"
# CLIENT_CONFIGS_0_MODEL="model-a"
# CLIENT_CONFIGS_0_PROXY_HTTP="http://user:pass@proxy.example.com:8080"
# CLIENT_CONFIGS_1_BASE_URL="http://another-api.com/v1"
# CLIENT_CONFIGS_1_API_KEY="sk-fghij"
# CLIENT_CONFIGS_1_MODEL="model-b"

evaluation_client_configs: List[ClientConfig] = []
for i in range(100):
    base_url = os.getenv(f"CLIENT_CONFIGS_{i}_BASE_URL")
    if not base_url:
        continue

    api_key = os.getenv(f"CLIENT_CONFIGS_{i}_API_KEY")
    model = os.getenv(f"CLIENT_CONFIGS_{i}_MODEL")
    
    if not api_key or not model:
        print(f"Warning: Incomplete configuration for CLIENT_CONFIGS_{i}. Skipping.")
        continue

    proxy = os.getenv(f"CLIENT_CONFIGS_{i}_PROXY_HTTP")

    evaluation_client_configs.append(ClientConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        proxy=proxy
    ))
    i += 1

# Fallback to default if no environment variables are set
if not evaluation_client_configs:
    DEFAULT_API_KEYS = os.getenv("DEFAULT_API_KEYS", "AIzaSyCoXKEaKODUXLZ7W6vhgm6jN6QDpvQc9PM").split(',')
    DEFAULT_BASE_URL = os.getenv("DEFAULT_BASE_URL", "http://103.228.38.165:1337/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "default")

    for key in DEFAULT_API_KEYS:
        if key:
            evaluation_client_configs.append(ClientConfig(
                base_url=DEFAULT_BASE_URL,
                api_key=key,
                model=DEFAULT_MODEL
            ))

client_rotator = None
if evaluation_client_configs:
    client_rotator = ClientRotator(evaluation_client_configs)
else:
    raise RuntimeError("No LLM client configurations found. Please set environment variables or provide defaults.")
