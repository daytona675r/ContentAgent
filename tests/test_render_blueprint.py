from pathlib import Path

import yaml


def test_render_blueprint_uses_public_owner_supplied_urls() -> None:
    blueprint = yaml.safe_load(Path("render.yaml").read_text())
    api, frontend = blueprint["services"]
    assert api["type"] == "web"
    assert frontend["type"] == "web"
    assert frontend["runtime"] == "static"
    assert "region" not in frontend
    assert frontend["rootDir"] == "frontend"
    assert frontend["staticPublishPath"] == "dist"
    assert frontend["envVars"] == [{"key": "VITE_API_BASE_URL", "sync": False}]
    assert api["envVars"][-1] == {"key": "PULSE_ALLOWED_ORIGINS", "sync": False}
