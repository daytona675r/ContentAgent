import json

from agent.demo_provider import FIXTURES, PLATFORMS, ReplayProvider


def test_fixture_copy_is_complete_and_platform_specific() -> None:
    for fixture in FIXTURES.values():
        provider = ReplayProvider(fixture)
        generated = json.loads(provider.generate(fixture.context).content)
        assert all(
            generated[platform]["post"] == fixture.drafts[platform] for platform in PLATFORMS
        )
        assert len({fixture.drafts[platform] for platform in PLATFORMS}) == 4
        assert all(fixture.refined[platform] != fixture.drafts[platform] for platform in PLATFORMS)
        assert all("context-first draft" not in fixture.drafts[platform] for platform in PLATFORMS)


def test_judging_is_stage_aware_and_stable() -> None:
    fixture = next(iter(FIXTURES.values()))
    first = ReplayProvider(fixture)
    second = ReplayProvider(fixture)
    for provider in (first, second):
        initial = provider.judge("twitter", fixture.drafts["twitter"]).content
        refined = provider.judge("twitter", fixture.refined["twitter"]).content
        assert '"score": 6.8' in initial
        assert '"score": 9.2' in refined
        assert provider.judge("twitter", fixture.drafts["twitter"]).content == initial
