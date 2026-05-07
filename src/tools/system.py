import subprocess
import os
import glob
import json
import re
from pathlib import Path
from src.tools.base import tool
from src.tools.registry import register_tool

@tool(name="bash", description="Execute a shell command.")
def bash(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = result.stdout
        if result.stderr:
            out += "\nSTDERR:\n" + result.stderr
        return out if out else "Command executed successfully with no output."
    except Exception as e:
        return f"Error executing bash: {e}"

@tool(name="powershell", description="Execute a PowerShell command (Windows only).")
def powershell(command: str) -> str:
    if os.name != "nt":
        return "Error: powershell tool is only available on Windows."
    try:
        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True, timeout=120)
        out = result.stdout
        if result.stderr:
            out += "\nSTDERR:\n" + result.stderr
        return out if out else "Command executed successfully with no output."
    except Exception as e:
        return f"Error executing powershell: {e}"

@tool(name="read_file", description="Read the contents of a file.")
def read_file(filepath: str) -> str:
    try:
        return Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file {filepath}: {e}"

@tool(name="write_file", description="Write content to a file, overwriting existing content.")
def write_file(filepath: str, content: str) -> str:
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file {filepath}: {e}"

@tool(name="edit_file", description="Edit a file by replacing old_text with new_text.")
def edit_file(filepath: str, old_text: str, new_text: str) -> str:
    try:
        content = Path(filepath).read_text(encoding="utf-8")
        if old_text not in content:
            return "Error: old_text not found in file."
        content = content.replace(old_text, new_text)
        Path(filepath).write_text(content, encoding="utf-8")
        return f"Successfully edited {filepath}"
    except Exception as e:
        return f"Error editing file {filepath}: {e}"

@tool(name="glob_search", description="Search for files using a glob pattern.")
def glob_search(pattern: str) -> str:
    try:
        files = glob.glob(pattern, recursive=True)
        return "\n".join(files) if files else "No files found."
    except Exception as e:
        return f"Error with glob search: {e}"

@tool(name="grep_search", description="Search for a regex pattern within files in a directory.")
def grep_search(pattern: str, directory: str = ".") -> str:
    try:
        regex = re.compile(pattern)
        results = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if regex.search(line):
                                results.append(f"{filepath}:{i+1}:{line.strip()}")
                except:
                    pass
        return "\n".join(results) if results else "No matches found."
    except Exception as e:
        return f"Error with grep search: {e}"

@tool(name="notebook_edit", description="Modify a Jupyter Notebook (.ipynb) file.")
def notebook_edit(filepath: str, cell_index: int, new_source: str) -> str:
    try:
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        if cell_index < 0 or cell_index >= len(cells):
            return "Error: Cell index out of bounds."
        cells[cell_index]["source"] = [new_source]
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return f"Successfully edited notebook {filepath} at cell {cell_index}"
    except Exception as e:
        return f"Error editing notebook: {e}"

for t in [bash, powershell, read_file, write_file, edit_file, glob_search, grep_search, notebook_edit]:
    register_tool(t)
