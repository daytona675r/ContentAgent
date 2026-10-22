#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL to the deployed FastAPI service URL}"
: "${FRONTEND_URL:?Set FRONTEND_URL to the deployed static-site URL}"

curl --fail --silent --show-error "${API_URL%/}/health"
printf '\nAPI health passed\n'
curl --fail --silent --show-error "${API_URL%/}/api/demo/scenarios" >/tmp/pulse-scenarios.json
SCENARIO_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["id"])' /tmp/pulse-scenarios.json)"
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "{\"scenario_id\":\"${SCENARIO_ID}\"}" \
  "${API_URL%/}/api/demo/run" >/tmp/pulse-demo.json
python -c 'import json; data=json.load(open("/tmp/pulse-demo.json")); assert data["demo"] is True; assert data["final_accepted"]'
curl --fail --silent --show-error "${FRONTEND_URL%/}/" >/dev/null
printf 'Frontend and representative demo flow passed.\n'
