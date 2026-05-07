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

```bash
# Scaffold project runtime directory
coalyx init

# Set API keys
coalyx config set gemini-api-key "YOUR_GEMINI_KEY"
coalyx config set openai-api-key "YOUR_OPENAI_KEY"

# Set Ollama (Optional)
coalyx config set ollama-api-base "http://localhost:11434"

# Start chatting (Instant mode)
coalyx chat

# Start chatting (Adaptive Reasoning mode)
coalyx chat --mode adaptive --model gemini/gemini-2.0-flash

# Using local Ollama models
coalyx chat --model ollama/gemma4:26b
```

## Commands

| Command | Description |
|---|---|
| `coalyx init` | Scaffold `.coalyx/` directory and `COALYX.md` |
| `coalyx config set <key> <value>` | Set API keys or config values |
| `coalyx config show` | Show current configuration |
| `coalyx chat` | Start interactive chat |
| `coalyx sessions` | List saved sessions |

## Chat Slash Commands

| Command | Description |
|---|---|
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
