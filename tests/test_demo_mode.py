from demo_mode import (
    DEMO_CONTEXT,
    DEMO_NOTICE,
    DEMO_OUTPUTS,
    DEMO_REFINEMENT,
    DEMO_SCORES,
)


def test_demo_payload_is_deterministic_and_labelled():
    assert "No OpenAI" in DEMO_NOTICE
    assert DEMO_CONTEXT
    assert set(DEMO_OUTPUTS) == {"LinkedIn", "Twitter", "Facebook", "Instagram"}
    assert all(0 <= score <= 10 for score in DEMO_SCORES.values())
    assert (
        DEMO_REFINEMENT["initial"]
        and DEMO_REFINEMENT["feedback"]
        and DEMO_REFINEMENT["accepted"]
    )
