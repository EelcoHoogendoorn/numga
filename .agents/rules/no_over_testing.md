# Testing Rules

- Always run targeted tests: run only the specific test file or target directly affected by your changes (e.g. `pytest tests/examples/geometry/test_multiview.py`).
- Only run the full test suite when editing test infrastructure across the entire suite or during explicit pre-commit checks.
