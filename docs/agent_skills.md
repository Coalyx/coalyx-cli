# Coalyx Agent Skills

Coalyx allows you to extend the AI's capabilities and behavior by defining **Agent Skills**. Skills are specialized instructions that are automatically injected into the system prompt when specific keywords or patterns are detected in your input.

This modular system allows you to transform Coalyx into an expert in various domains (e.g., Data Science, DevOps, Creative Writing) without cluttering the global system prompt.

---

## 1. How it Works

1.  **Discovery**: On startup, Coalyx scans the `~/.coalyx/skills/` directory for any files ending in `.md`.
2.  **Parsing**: Each skill file must follow a specific format with **YAML frontmatter** containing the skill's name and trigger patterns.
3.  **Matching**: During a chat session, Coalyx checks every user message against the registered triggers.
4.  **Activation**: If a match is found (case-insensitive), Coalyx displays a "Skill activated" notification and appends the skill's instructions to the session's system context for the remainder of the conversation.

---

## 2. Skill File Format

A skill is defined in a Markdown file with the following structure:

```markdown
---
name: Skill Name
triggers: keyword1, keyword2, regex.*pattern
---
Instructions to be injected into the system prompt.
```

### Components:
-   **Frontmatter (`---`)**:
    -   `name`: A descriptive name for the skill.
    -   `triggers`: A comma-separated list of strings or regular expressions. If any of these match the user's input, the skill is triggered.
-   **Body**:
    -   The text below the second `---` is the actual instruction set. You can use this to define personas, coding standards, or specific domain knowledge.

---

## 3. Practical Examples

### Data Science Expert
**File**: `~/.coalyx/skills/data_science.md`
```markdown
---
name: Data Science Expert
triggers: pandas, dataframe, numpy, scikit-learn, plot, visualization
---
You are a senior Data Scientist. When writing code:
- Use Pandas for data manipulation.
- Prefer Plotly for interactive visualizations.
- Always include brief explanations of the statistical methods used.
- Ensure all plots have proper labels and titles.
```

### Git & DevOps Assistant
**File**: `~/.coalyx/skills/devops.md`
```markdown
---
name: DevOps Assistant
triggers: docker, kubernetes, terraform, ci/cd, workflow, git
---
You are an expert DevOps engineer. 
- Focus on security, idempotency, and best practices in infrastructure as code.
- When suggesting Dockerfiles, use multi-stage builds and non-root users.
- For Git commands, explain exactly what each flag does.
```

---

## 4. Management & Troubleshooting

-   **Adding Skills**: Simply drop a new `.md` file into `~/.coalyx/skills/` and restart your session.
-   **Overlapping Triggers**: If multiple skills match a single input, only the first one found will be activated for that turn (to prevent context bloat).
-   **Case Sensitivity**: Triggers are matched using case-insensitive regex.
-   **Location**: If the `~/.coalyx/skills/` directory does not exist, you can create it manually or it will be created automatically on the first run of Coalyx.
