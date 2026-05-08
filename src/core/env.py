import os
import sys
import venv
import subprocess
import shutil
from pathlib import Path
from rich.console import Console

console = Console()

REQUIRED_PACKAGES = [
    "numpy", "scipy", "pandas", "polars", "pyarrow", "sympy",
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
    
    # Install packages
    with console.status("[bold cyan]Installing packages...[/bold cyan]", spinner="dots"):
        try:
            # Upgrade pip first
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
                check=True, capture_output=True
            )
            
            # Install all required packages
            subprocess.run(
                [str(python_exe), "-m", "pip", "install"] + REQUIRED_PACKAGES,
                check=True, capture_output=True
            )
            console.print("[bold green]Global environment ready![/bold green]")
        except subprocess.CalledProcessError as e:
            console.print("[bold red]Failed to install packages.[/bold red]")
            if e.stderr:
                console.print(f"[dim]{e.stderr.decode()}[/dim]")
            # Cleanup broken venv
            shutil.rmtree(venv_dir, ignore_errors=True)
