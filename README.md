# Coalyx CLI

A high-performance, multi-task AI Chat Terminal featuring **Adaptive Uncertainty-Aware Reasoning**. Coalyx isn't just another chat interface; it's an intelligent agent framework designed for complex problem solving and autonomous task execution.

---

## Key Features

### Intelligent Adaptive Reasoning
Coalyx utilizes a sophisticated reasoning engine that evaluates complex queries through multiple internal perspectives. It autonomously identifies potential inconsistencies and performs real-time self-correction to ensure the most reliable and accurate output for mission-critical tasks.

### High-Fidelity Session Memory
Maintain perfect project awareness through an advanced context management architecture. Coalyx intelligently monitors interaction density and employs specialized state-compaction techniques to preserve long-term project knowledge while staying within model processing limits.

### Unified Extension Ecosystem
Boost your productivity with native support for system-level tools, real-time web intelligence, and external service orchestration via the Model Context Protocol (MCP). The modular skill system allows for seamless injection of custom domain-specific expertise.

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
│   ├── sessions/          # Saved chat history
│   └── skills/            # Custom agent instructions
└── src/
    ├── core/              # Reasoning pipeline & embedding logic
    ├── tools/             # Built-in functional tools
    └── cli/               # Rich UI & Command handling
```

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.