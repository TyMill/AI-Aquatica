# Contributing to AI-Aquatica

AI-Aquatica welcomes contributions that improve reproducibility, water-quality workflows, statistical methods, machine-learning helpers, visualization, and documentation.

## Development setup

```bash
git clone https://github.com/TyMill/AI-Aquatica.git
cd AI-Aquatica
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[testing,interactive]"
```

## Quality checks

Before opening a pull request, run:

```bash
python -m pytest
python -m compileall ai_aquatica
```

Optional dependency groups are intentionally separated. Core functionality must import without TensorFlow, Plotly, MongoDB, OpenPyXL, SQLAlchemy, or Requests.

## Pull request expectations

1. Add or update tests for new behaviour.
2. Keep examples deterministic where possible.
3. Document public functions in `docs/` or `README.md`.
4. Avoid hidden external services in tests; mock APIs and databases.
5. Preserve backward-compatible public APIs unless a breaking change is clearly justified.
