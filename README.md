# Coalyx CLI

A powerful, multi-task AI Chat CLI featuring **Adaptive Uncertainty-Aware Reasoning**.

## Features

- **Multi-Model Support**: Use models from OpenAI, Google Gemini, Ollama via `litellm`.
- **Instant Mode**: Fast, direct responses for basic tasks.
- **Adaptive Reasoning Mode**: Samples multiple reasoning paths, measures semantic uncertainty via Gemini Embeddings, and triggers Self-Doubt reflection when the model is uncertain.
- **Memory Management**: Context budget tracking with 4 zones, conversation compaction (`/compact`), and session persistence.
- **Project Memory**: `COALYX.md` as a persistent project knowledge anchor loaded into every session.
- **Extension Events**: Skills auto-activate on task match. Hooks fire on lifecycle events (SessionStart, SessionEnd, PreCommit, etc.).
- **Rich Dashboard**: Real-time monitoring of tokens, context usage, generation speed, and memory zone.

## Installation

```bash
git clone <this_repo>
cd coalyx-cli
pip install -e .
```

## Quick Start

Just run a single command to get started:

```bash
coalyx
```

If it's your first time running the CLI, Coalyx will automatically scaffold the `.coalyx/` directory and display an interactive **Setup Panel** right in your terminal so you can configure your API keys (you only need to do this once).

Whenever you want to update your API keys, just type the following command directly in the chat interface:
```text
/config
```

### Advanced Run Options

```bash
# Enable deep reasoning (Adaptive Reasoning mode)
coalyx --mode adaptive --model gemini/gemini-2.0-flash

# Use a local model via Ollama
coalyx --model ollama/gemma4:26b
```

## Commands

| Command | Description |
|---|---|
| `coalyx` | Launch the main interactive chat interface |
| `coalyx --mode adaptive` | Run Coalyx with Adaptive Reasoning mode |
| `coalyx --help` | View all launch parameters |

## Chat Slash Commands

| Command | Description |
|---|---|
| `/config` | Open the Interactive Setup Panel to change API keys |
| `/model <name>` | Switch the active AI model dynamically |
| `/compact` | Compress conversation history |
| `/clear` | Clear session and reload project memory |
| `/mode` | Toggle Instant / Adaptive Reasoning |
| `/status` | Show session dashboard |
| `/help` | Show available commands |
| `/quit` | Exit the chat |

## Project Structure

```
coalyx-cli/
├── .mcp.json              # MCP server connections
├── COALYX.md              # Project memory anchor
├── .coalyx/
│   ├── settings.local.json
│   ├── sessions/
│   └── skills/
└── src/
    ├── core/              # Pipeline, model, config, monitor, embeddings
    ├── memory/            # Context tracking, compaction, session store
    ├── extensions/        # Skills, hooks, registry
    └── cli/               # UI and entry point
```

## License

MIT License — see [LICENSE](LICENSE) for details.
