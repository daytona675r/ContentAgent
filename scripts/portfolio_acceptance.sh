#!/usr/bin/env bash
set -euo pipefail

echo "== PULSE portfolio acceptance =="

# The maintained application, not only demo fixtures, is part of the quality gate.
ruff check src tests app.py demo_mode.py
ruff format --check src tests app.py demo_mode.py

python -m pytest -q

# Execute imports from the installed real package.
python -c "from agent.social_content_graph import social_content_graph; print(type(social_content_graph).__name__)"

# Demo bootstrap must remain provider-credential free.
DEMO_MODE=1 python -c "import demo_mode; assert demo_mode.DEMO_REFINEMENT and demo_mode.DEMO_OUTPUTS"

python - <<'PY'
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
assert "Python-3.11" in readme or "Python 3.11" in readme, "README must advertise the supported Python 3.11+ runtime"
assert "git clone <https://github.com/daytona675r/ContentAgent>" not in readme, "Clone command still contains invalid angle brackets"
PY

echo "PULSE acceptance passed."
