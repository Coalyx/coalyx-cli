import os
import sys
import venv
import subprocess
import shutil
import time
from pathlib import Path
from rich.console import Console

console = Console()

REQUIRED_PACKAGES = [
    "numpy", "scipy", "pandas", "pyarrow", "sympy",
    "statsmodels", "scikit-learn", "matplotlib", "seaborn",
    "plotly", "networkx", "numba", "openpyxl", "xlsxwriter",
    "python-dateutil", "pytz", "tqdm", "requests",
    "beautifulsoup4", "lxml", "pydantic", "jupyterlab"
]

def get_venv_dir(coalyx_dir: Path) -> Path:
    """Return the path to the global virtual environment."""
    return coalyx_dir / "venv"

def get_venv_bin_dir(coalyx_dir: Path) -> Path:
    """Return the bin/Scripts directory of the virtual environment."""
    venv_dir = get_venv_dir(coalyx_dir)
    if os.name == "nt":
        return venv_dir / "Scripts"
    return venv_dir / "bin"

def get_venv_python(coalyx_dir: Path) -> Path:
    """Return the path to the python executable inside the virtual environment."""
    bin_dir = get_venv_bin_dir(coalyx_dir)
    exe = "python.exe" if os.name == "nt" else "python"
    return bin_dir / exe

def setup_global_venv(coalyx_dir: Path) -> None:
    """Create the virtual environment and install required packages if it doesn't exist."""
    venv_dir = get_venv_dir(coalyx_dir)
    if venv_dir.exists():
        return
        
    console.print("[bold yellow]Initializing Coalyx global environment...[/bold yellow]")
    console.print("[dim]This may take a few minutes as data science packages are installed.[/dim]")
    
    # Create venv
    try:
        venv.create(venv_dir, with_pip=True)
    except Exception as e:
        console.print(f"[bold red]Failed to create virtual environment:[/bold red] {e}")
        return

    python_exe = get_venv_python(coalyx_dir)
    
    # Bootstrap 'uv' into the venv for maximum speed and reliability
    console.print("[bold cyan]Bootstrapping environment...[/bold cyan]")
    
    # Try to install uv into the venv first
    with console.status("[dim]Installing 'uv' bootstrap...[/dim]", spinner="dots"):
        try:
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "uv", "--no-input"],
                check=True, capture_output=True
            )
            has_uv = True
        except Exception:
            has_uv = False

    # Find the uv binary inside the venv
    bin_dir = get_venv_bin_dir(coalyx_dir)
    uv_exe = bin_dir / ("uv.exe" if os.name == "nt" else "uv")

    # Install packages
    console.print("[bold cyan]Installing packages...[/bold cyan]")
    try:
        if has_uv and uv_exe.exists():
            console.print("[dim](Using 'uv' bootstrap for 10x faster installation)[/dim]")
            subprocess.run(
                [str(uv_exe), "pip", "install"] + REQUIRED_PACKAGES,
                check=True
            )
        else:
            # Fallback to standard pip
            console.print("[dim](Bootstrap failed, falling back to standard pip)[/dim]")
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "--no-input"],
                check=True
            )
            subprocess.run(
                [str(python_exe), "-m", "pip", "install"] + REQUIRED_PACKAGES + ["--no-input"],
                check=True
            )
        console.print("[bold green]Global environment ready![/bold green]")
        time.sleep(1)
        console.clear()
    except subprocess.CalledProcessError as e:
        console.print("[bold red]Failed to install packages.[/bold red]")
        # Cleanup broken venv
        shutil.rmtree(venv_dir, ignore_errors=True)
