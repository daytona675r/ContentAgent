"""Authored deterministic responses for the public PULSE demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

PLATFORMS = ("twitter", "facebook", "linkedin", "instagram")


@dataclass(frozen=True)
class ScenarioFixture:
    id: str
    label: str
    context: str
    drafts: dict[str, str]
    refined: dict[str, str]
    feedback: dict[str, str]


SCENARIO_FIXTURES = (
    ScenarioFixture(
        id="onboarding-saas",
        label="B2B onboarding SaaS · operations teams",
        context=(
            "Fictional sample: Launchpath helps operations teams turn scattered onboarding "
            "checklists into a shared, reviewable workspace. The brief is to reach teams "
            "losing handoff context and invite them to try a guided workspace."
        ),
        drafts={
            "twitter": "Onboarding gets easier when the context stays together. See what Launchpath can do.",
            "linkedin": "Onboarding teams need better context. Launchpath brings the work together so teams can move forward. Learn more.",
            "facebook": "Starting a new customer should feel organized, not scattered. Discover a simpler onboarding workspace with Launchpath.",
            "instagram": "From first handoff to first win, keep the onboarding story in one place. Explore Launchpath.",
        },
        refined={
            "twitter": "Ops teams: stop losing onboarding handoffs in scattered docs. Launchpath keeps checklists and context together—try the guided workspace.",
            "linkedin": "When onboarding handoffs live in separate docs, operations teams spend time reconstructing context. Launchpath gives the team one reviewable workspace for checklists, owners and next steps. See the guided workflow.",
            "facebook": "For operations teams, a new customer should not mean hunting through old checklists. Launchpath keeps owners, handoffs and next steps together in one guided workspace. Take a look at the sample workflow.",
            "instagram": "A clearer onboarding handoff starts with one shared place for the checklist, context and next step. Launchpath is designed for ops teams—save the guided workspace for your next rollout.",
        },
        feedback={
            "twitter": "Name the operations audience and replace the broad benefit with a concrete handoff problem and CTA.",
            "linkedin": "Make the operational pain and the reviewable workspace explicit, then give readers one clear next step.",
            "facebook": "Speak directly to operations teams and show what stays together instead of using a general promise.",
            "instagram": "Add a visual-friendly sequence and a specific audience so the CTA has a clear reason to act.",
        },
    ),
    ScenarioFixture(
        id="repairable-commuter",
        label="Repairable commuter product · everyday riders",
        context=(
            "Fictional sample: Switchback makes a repairable commuter light with replaceable "
            "battery and lens modules. The brief is to reach everyday riders who want a "
            "dependable light they can maintain instead of discard."
        ),
        drafts={
            "twitter": "A commuter light made for the everyday ride. Meet Switchback and see the difference.",
            "linkedin": "Switchback takes a different approach to commuter lighting: build a product people can keep using. Explore the repairable design.",
            "facebook": "Your daily ride deserves a light you can count on. Meet Switchback, designed with commuters in mind.",
            "instagram": "Ride after ride, keep the light going. Switchback brings a repairable approach to the commute. Learn more.",
        },
        refined={
            "twitter": "Daily riders: replace the battery or lens instead of the whole light. Switchback is a repairable commuter light—see the module design.",
            "linkedin": "For everyday riders, a dead battery should not make a whole commuter light disposable. Switchback separates replaceable battery and lens modules from the housing. Explore the repairable design.",
            "facebook": "If your commute depends on a light, you should be able to keep it working. Switchback lets everyday riders replace the battery or lens module instead of throwing the whole light away. See how it is made.",
            "instagram": "Battery low? Lens scratched? Keep the ride moving with replaceable modules. Switchback is a repairable commuter light for everyday riders—tap through to see the design.",
        },
        feedback={
            "twitter": "Identify everyday riders and name the replaceable parts so the repairability benefit is concrete.",
            "linkedin": "Connect the design choice to the commuter problem and specify what can be replaced.",
            "facebook": "Make the practical maintenance benefit tangible and keep the community-friendly invitation.",
            "instagram": "Use a visual repair moment and a direct commuter CTA rather than a broad sustainability claim.",
        },
    ),
    ScenarioFixture(
        id="neighborhood-night",
        label="Local evening event · neighborhood residents",
        context=(
            "Fictional sample: Night Market North is a fictional neighborhood evening event "
            "with local food stalls, small makers and a repair corner. The brief is to invite "
            "nearby residents to plan an easy, welcoming Friday visit."
        ),
        drafts={
            "twitter": "Friday night has a new local plan. Join Night Market North and bring a friend.",
            "linkedin": "Night Market North brings local makers and neighbors together for an evening event. Find out what is planned for Friday.",
            "facebook": "Looking for a relaxed Friday night close to home? Night Market North is bringing food, makers and neighbors together.",
            "instagram": "Friday lights, local makers and something good to eat. Night Market North is nearly here—come along.",
        },
        refined={
            "twitter": "Northside neighbors: plan an easy Friday at Night Market North—local food stalls, makers and a repair corner. See the visit guide and bring a friend.",
            "linkedin": "Night Market North gives neighborhood residents a simple Friday plan: meet local makers, eat from nearby stalls and bring a small item to the repair corner. View the evening guide.",
            "facebook": "Neighbors, your Friday plan is here: Night Market North has local food stalls, small makers and a repair corner for the community. Check the evening guide and invite someone nearby.",
            "instagram": "A Friday evening made for the neighborhood: food stalls, maker tables and a repair corner under the lights. Save the Night Market North guide and bring a friend.",
        },
        feedback={
            "twitter": "Name the local audience and list the event elements so the invitation feels specific and actionable.",
            "linkedin": "Clarify who the event is for and turn the broad announcement into a useful visit plan.",
            "facebook": "Add the concrete activities and a nearby-neighbor invitation to make the community value clear.",
            "instagram": "Give the visual scene more detail and make saving the guide the explicit CTA.",
        },
    ),
)

FIXTURES = {fixture.id: fixture for fixture in SCENARIO_FIXTURES}


@dataclass
class ReplayProvider:
    """Stable fixture-backed responses passed through the real graph."""

    fixture: ScenarioFixture

    def _message(self, payload: object) -> SimpleNamespace:
        return SimpleNamespace(content=json.dumps(payload), usage_metadata={"total_tokens": 0})

    def generate(self, _: str) -> SimpleNamespace:
        return self._message(
            {platform: {"post": self.fixture.drafts[platform]} for platform in PLATFORMS}
        )

    def judge(self, platform: str, content: str) -> SimpleNamespace:
        is_refined = content.strip() == self.fixture.refined[platform]
        score = 9.2 if is_refined else 6.8
        feedback = (
            "Accepted: the audience, concrete value and next step are clear."
            if is_refined
            else self.fixture.feedback[platform]
        )
        return self._message({"strategist": {"score": score, "feedback": feedback}})

    def refine(self, platform: str, _: str, __: str) -> SimpleNamespace:
        return self._message({"post": self.fixture.refined[platform]})
