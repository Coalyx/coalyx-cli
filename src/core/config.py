import json
import os
from pathlib import Path
from typing import Dict, Any

CONFIG_FILE = Path.home() / ".coalyx_config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from the config file."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_config(config_data: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific configuration value."""
    config = load_config()
    return config.get(key, default)

def set_config_value(key: str, value: Any) -> None:
    """Set a specific configuration value."""
    config = load_config()
    config[key] = value
    save_config(config)

def setup_environment() -> None:
    """Inject configuration values into environment variables for litellm."""
    config = load_config()
    if "openai-api-key" in config:
        os.environ["OPENAI_API_KEY"] = config["openai-api-key"]
    if "gemini-api-key" in config:
        # litellm requires GEMINI_API_KEY; google SDK uses GOOGLE_API_KEY
        os.environ["GEMINI_API_KEY"] = config["gemini-api-key"]
        os.environ["GOOGLE_API_KEY"] = config["gemini-api-key"]
    if "ollama-api-base" in config:
        os.environ["OLLAMA_API_BASE"] = config["ollama-api-base"]
