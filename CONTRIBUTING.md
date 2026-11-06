# Contributing to FreightIQ

Thank you for considering a contribution. This project has a small surface area and strict safety properties, so the contribution process is intentionally short.

## Getting started

```bash
git clone https://github.com/yyouretoast/freightiq.git
cd freightiq
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your GROQ_API_KEY (or configure OpenAI/Ollama)
python scripts/seed_db.py       # idempotent DB + vector index seeding
streamlit run app.py
```

## Before opening a PR

Run all four verification commands from `README.md` (§ Verification Test Suite) and paste the output into the PR description:

```bash
python -m tests.verify_system
python -m tests.evaluate_retrieval
python -m tests.evaluate_agent_trajectories
python -m tests.stress_test_concurrency
```

## Ground rules

- **Style / lint**: follow the `ruff` configuration in `pyproject.toml`.
- **Database access**: the SQL tool must stay `SELECT`-only, wrapped in a bounded subquery (`LIMIT 25`), and opened via `file:DB?mode=ro`. Never introduce non-standard SQLite initialization outside `scripts/seed_db.py`.
- **Fallbacks are load-bearing**: preserve the CrossEncoder cosine fallback (`rag/reranker.py`), the DuckDuckGo fallback (`agent/tools.py`), and the sibling-model failover (`agent/nodes.py`).
- **Guardrails are load-bearing**: keep turn-scoped loop detection and the 8-message context window intact (see `AGENTS.md`).
- **One scope per PR**: keep changes focused; large refactors are best discussed first in an issue.

## Commit / PR style

- Prefer [Conventional Commits](https://www.conventionalcommits.org/) prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).
- Reference related issues with `Closes #N` in the PR body.
- Update `README.md`, the relevant ADR in `docs/adr/`, or `DATA.md` if your change alters documented behavior, benchmarks, or schema.

## Questions

Open an [issue](https://github.com/yyouretoast/freightiq/issues) — feature requests and design questions go there before code.
