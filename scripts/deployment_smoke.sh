#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL to the deployed FastAPI service URL}"
: "${FRONTEND_URL:?Set FRONTEND_URL to the deployed static-site URL}"

curl --fail --silent --show-error "${API_URL%/}/health"
printf '\nAPI health passed\n'
curl --fail --silent --show-error "${API_URL%/}/api/demo/scenarios" >/tmp/pulse-scenarios.json
read -r SCENARIO_ID OTHER_SCENARIO_ID <<EOF
$(python -c 'import json; ids=[item["id"] for item in json.load(open("/tmp/pulse-scenarios.json"))]; print(ids[0], ids[1])')
EOF
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "{\"scenario_id\":\"${SCENARIO_ID}\"}" \
  "${API_URL%/}/api/demo/run" >/tmp/pulse-demo.json
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "{\"scenario_id\":\"${OTHER_SCENARIO_ID}\"}" \
  "${API_URL%/}/api/demo/run" >/tmp/pulse-demo-other.json
python - <<'PY'
import json
first = json.load(open("/tmp/pulse-demo.json"))
second = json.load(open("/tmp/pulse-demo-other.json"))
assert first["demo"] is True and first["final_accepted"]
assert first["scenario"]["id"] != second["scenario"]["id"]
assert first["final_accepted"]["twitter"] != second["final_accepted"]["twitter"]
for data in (first, second):
    assert all(len(history) >= 2 for history in data["refinement_history"].values())
    text = " ".join(data["final_accepted"].values()).lower()
    assert "context-first draft" not in text
    assert "refined, context-first" not in text
PY
curl --fail --silent --show-error "${FRONTEND_URL%/}/" >/dev/null
printf 'Frontend and representative demo flow passed.\n'
