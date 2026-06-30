# Contributing to SolarPro

Thanks for your interest! SolarPro is an open-source portfolio project and
contributions, issues, and suggestions are welcome.

## Getting started

```bash
git clone https://github.com/gaiaflaviamezaib/SolarPro.git
cd SolarPro
pip install -e ".[dev]"
pytest          # run the test suite
ruff check .    # lint
```

## Guidelines

- **Keep the core dependency-light.** The `solarpro` package must remain
  Pyodide-compatible so it keeps running in the browser. Avoid adding heavy or
  non-WASM dependencies to the core; put optional tooling under
  `pyproject.toml` extras instead.
- **Match the existing style.** Type hints, module-level docstrings, and `ruff`
  formatting (line length 100). Cite a reference for any new physical model.
- **Add tests** for new behavior in `tests/`.
- **Don't invent results.** This repository values accuracy: model constants
  should be cited, and sample data is clearly labelled as synthetic.

## Pull requests

1. Fork and create a feature branch.
2. Ensure `pytest` and `ruff check .` pass.
3. Describe the change and its motivation in the PR.

## Reporting issues

Open a GitHub issue with steps to reproduce, expected vs. actual behavior, and
your environment (OS, Python version).
