import json
import os
from pathlib import Path
from typing import Dict, Any

CONFIG_FILE = Path.home() / ".coalyx_config.json"

DEFAULT_CONFIG = {
    "reasoning": {
        "mode": "adaptive",
        "max_candidates": 3,
        "max_tool_rounds": 5,
        "ask_before_research": False,
        "ask_before_file_write": True,
        "ask_before_shell": True,
        "auto_research_on_uncertainty": True,
        "show_uncertainty_summary": True,
        "show_assumptions": True,
    },
    "adaptive_paths": {
        "enabled": False,
        "min_paths": 1,
        "default_paths": 3,
        "max_paths": 6,
        "max_paths_per_wave": 3,
        "max_waves": 3,
        "target_confidence": 0.78,
        "min_marginal_gain": 0.08,
        "max_total_path_tokens": 8000,
        "max_final_tokens": 1000,
    },
}

def load_config() -> Dict[str, Any]:
    """Load configuration from the config file."""
    config = DEFAULT_CONFIG.copy()
    if not CONFIG_FILE.exists():
        return config
    try:
        with open(CONFIG_FILE, "r") as f:
            user_config = json.load(f)
            # Simple merge
            for k, v in user_config.items():
                if k == "reasoning" and isinstance(v, dict):
                    config["reasoning"].update(v)
                else:
                    config[k] = v
            return config
    except json.JSONDecodeError:
        return config

def save_config(config_data: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)


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
