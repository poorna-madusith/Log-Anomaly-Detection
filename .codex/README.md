# Project Codex agents

This project includes three project-scoped Codex subagents in `agents/`:

- **AI Engineer** (`ai-engineer.toml`) — model development, training, inference, and ML deployment work. Adapted from [Agency Agents](https://github.com/msitarzewski/agency-agents/blob/main/engineering/engineering-ai-engineer.md) for this repository.
- **ML Evaluation and Statistics** (`ml-evaluation-statistics.toml`) — experimental validity, leakage, drift comparisons, calibration, uncertainty, and research claims.
- **Data Pipeline Engineer** (`data-pipeline-engineer.toml`) — raw log ingestion, labels, Drain fitting, token conventions, splits, sampling, and preprocessing reproducibility.

Ask Codex to use the **AI Engineer** for model or inference work, **ML Evaluation and Statistics** for experiment reviews, and **Data Pipeline Engineer** for ingestion or preprocessing work. Each configuration omits a model override, so it inherits the active Codex session's model settings. These agents do not change user-global configuration.

Codex loads project custom agents from `.codex/agents/`. If the agent is not available in an already-open session, start a new project session after adding the file.
