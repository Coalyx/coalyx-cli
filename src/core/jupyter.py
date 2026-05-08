import os
import subprocess
import time
import json
import secrets
from pathlib import Path
from rich.console import Console
from src.core.env import get_venv_python

console = Console()

def get_jupyter_server_info(coalyx_dir: Path):
    """Check if a jupyter server is running and return its (url, token)."""
    python_exe = get_venv_python(coalyx_dir)
    try:
        result = subprocess.run(
            [str(python_exe), "-m", "jupyter", "server", "list", "--json"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                return data.get('url'), data.get('token')
    except Exception:
        pass
    return None, None

def start_jupyter_server(coalyx_dir: Path):
    """Start jupyter lab in the background and return (url, token)."""
    python_exe = get_venv_python(coalyx_dir)
    token = secrets.token_hex(16)
    port = 8888
    log_file = coalyx_dir / "jupyter.log"
    
    console.print(f"[dim]Starting Jupyter Lab on port {port}...[/dim]")
    
    cmd = [
        str(python_exe), "-m", "jupyter", "lab",
        "--no-browser",
        f"--port={port}",
        f"--IdentityProvider.token={token}",
        "--ip=127.0.0.1"
    ]
    
    with open(log_file, "w") as f:
        subprocess.Popen(
            cmd,
            stdout=f,
            stderr=f,
            start_new_session=True
        )
    
    for _ in range(15):
        time.sleep(1)
        url, active_token = get_jupyter_server_info(coalyx_dir)
        if url:
            return url.rstrip('/'), active_token
            
    return f"http://localhost:{port}", token

def ensure_jupyter_running(coalyx_dir: Path):
    """Ensure jupyter is running and return (url, token)."""
    url, token = get_jupyter_server_info(coalyx_dir)
    if url:
        return url.rstrip('/'), token
    
    return start_jupyter_server(coalyx_dir)
