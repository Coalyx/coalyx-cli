# Coalyx CLI

A lightweight, terminal-based AI chat application designed for seamless interactions with multiple LLM providers. Much like the Ollama CLI, Coalyx provides a clean and distraction-free environment for chatting directly from your terminal. It is built to be a smart conversational assistant rather than an autonomous coding agent.

---

## Key Features

- **Multi-Provider Support**: Chat with models from OpenAI, Anthropic, Google Gemini, and local models via Ollama.
- **Adaptive Reasoning**: Employs a multi-path evaluation pipeline (Direct, Skeptical, and Context-aware) to resolve uncertainty and ensure logical consistency.
- **Smart Context Management**: Automatically manages long conversations via intelligent compaction without exceeding token limits.
- **Session Workspace**: Every chat is a sandboxed environment with persistent artifacts, tool logs, and automated secret redaction.
- **Tool Integration**: Enhances conversations with built-in utilities like web searching, math evaluation, and file reading.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Coalyx/coalyx-cli.git
cd coalyx-cli

# Install in editable mode
pip install -e .
```

---

## Quick Start

Simply run the main command:

```bash
coalyx
```

### First-Run Experience
If it's your first time, Coalyx will:
1. Automatically scaffold the `.coalyx/` project directory.
2. Launch an **Interactive Setup Panel** to configure your API keys.
3. *Note*: Adaptive mode requires a Gemini API key for embeddings. Get one here: [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys).

### Runtime Options

```bash
# Start with Adaptive Reasoning enabled
coalyx --mode adaptive

# Use a specific model (OpenAI, Gemini, or local Ollama)
coalyx --model gpt-4o
coalyx --model ollama/llama3
```

---

## Chat Commands

| Command | Action |
|:---|:---|
| `/config` | Open Setup Panel to update API keys/Settings |
| `/model <name>` | Switch the active model on-the-fly |
| `/mode` | Toggle between **Instant** and **Adaptive** reasoning |
| `/compact` | Compress conversation history into a summary |
| `/file <path>` | Inject the contents of a file into your prompt |
| `/status` | View a detailed token & speed dashboard |
| `/clear` | Reset the session while keeping project memory |
| `/quit` | Save and exit |

---

## Project Structure

```text
.
├── COALYX.md              # Project-wide persistent memory
├── .mcp.json              # MCP Server configuration
├── .coalyx/               # Runtime state (git-ignored)
│   ├── settings.json      # Local configuration
│   ├── sessions/          # Structured Reasoning Workspaces
│   │   └── <session_id>/  # Manifest, artifacts, and logs
│   └── skills/            # Custom agent instructions
└── src/
    ├── core/              # Reasoning pipeline & embedding logic
    ├── tools/             # Built-in functional tools
    ├── memory/            # Session Workspace & Context management
    └── cli/               # Rich UI & Command handling
```

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.