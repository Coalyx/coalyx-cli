# Coalyx CLI User Guide

Coalyx is a lightweight, terminal-based AI chat application designed to provide a seamless and distraction-free interaction with multiple LLM providers directly from your terminal. 

This guide covers everything you need to know to install, configure, and use Coalyx effectively.

## 1. Initial Setup

After installing the package via pip, you need to configure your API keys.

1. Run the main command:
   ```bash
   coalyx
   ```
2. On the first run, the CLI will prompt you for your API keys (Gemini, OpenAI, or Ollama local URL). You can leave a field blank to skip it if you do not plan to use that specific provider.
3. If you ever need to reconfigure your keys, use the `/config` slash command inside the application.

## 2. Core Concepts

Coalyx operates using two primary reasoning modes:

- **Instant Mode**: Passes your query directly to the LLM and returns the response immediately. Best for quick questions and casual conversation.
- **Adaptive Reasoning Mode**: Employs a multi-step evaluation pipeline. Coalyx generates multiple candidate responses internally, checks for consistency, and triggers self-correction if it detects conflicting information. Best for complex problem solving and logical tasks.

You can toggle between these modes at any time by typing `/mode` in the chat.

## 3. Session Management

Coalyx automatically saves every conversation session to disk. You do not need to manually save your progress.

- **Listing Sessions**: Inside the chat, type `/sessions` to view your 10 most recent sessions, including their Session IDs, the model used, and the last updated time.
- **Resuming a Session**: To pick up where you left off, launch the application using the `--resume` flag followed by the Session ID:
  ```bash
  coalyx --resume 718da76103c2
  ```

## 4. Context & Memory

Long conversations can exceed the model's token context window. Coalyx features a built-in session monitor to prevent this.

- The application displays a "Context" bar indicating your current token usage percentage.
- When the context reaches a critical level (>90%), Coalyx will automatically run a background compaction process. It summarizes older messages into a dense system prompt to free up token space while preserving the core facts of the conversation.
- You can manually trigger this compaction at any time using the `/compact` command.

## 5. Slash Commands

Coalyx includes several built-in commands to control the environment during a chat session.

| Command | Description | Example Usage |
|---------|-------------|---------------|
| `/model <name>` | Switch the active AI model on the fly. | `/model gemini/gemini-2.0-flash` |
| `/mode` | Toggle between Instant and Adaptive Reasoning modes. | `/mode` |
| `/sessions` | Display a list of your recently saved sessions. | `/sessions` |
| `/status` | View a dashboard showing memory, token usage, and speed. | `/status` |
| `/compact` | Manually compress the chat history to free up context space. | `/compact` |
| `/clear` | Clear the current conversation history and start fresh. | `/clear` |
| `/config` | Launch the setup wizard to update your API keys. | `/config` |
| `/help` | Show the quick reference menu for commands. | `/help` |
| `/quit` | Exit the application safely. | `/quit` |

### Multi-line Input

If you need to paste a block of text or write a long, multi-line message, use the `"""` block syntax:

```text
You: """
Here is a block of text.
It spans multiple lines.
Press Enter to continue adding lines.
"""
```
Alternatively, type `line\` at the end of a line to continue on the next line.
