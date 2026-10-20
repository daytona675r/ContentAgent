"""Credential-free deterministic PULSE portfolio demo."""

from __future__ import annotations

DEMO_NOTICE = "Demo mode: replayed example output. No OpenAI, Google or Tavily call is made."
DEMO_CONTEXT = "A small B2B product team wants to turn customer research into platform-specific content without losing business context."
DEMO_OUTPUTS = {
    "LinkedIn": "A practical content workflow should preserve the customer problem, audience and evidence before adapting a message to a platform.",
    "Twitter": "Good content starts with context. Capture the customer problem first, then adapt the message to the channel.",
    "Facebook": "Share the customer story, the problem it reveals and the next step readers can take.",
    "Instagram": "Context first. Channel second. Make the customer problem clear before polishing the post.",
}
DEMO_SCORES = {"Strategist": 8.6, "Customer": 8.2, "Industry expert": 8.4, "Investor": 7.9}
DEMO_REFINEMENT = {
    "initial": "A generic post about solving customer problems.",
    "feedback": "Make the customer context and practical next step explicit.",
    "accepted": DEMO_OUTPUTS["LinkedIn"],
}


def render_demo(st) -> None:
    st.set_page_config(page_title="PULSE · Demo", page_icon="🌐", layout="wide")
    st.title("🌐 PULSE")
    st.info(DEMO_NOTICE)
    st.caption("Representative sample inputs and replayed outputs for portfolio review.")
    st.subheader("Business context")
    st.write(DEMO_CONTEXT)
    st.subheader("Generated platform drafts")
    for platform, output in DEMO_OUTPUTS.items():
        with st.expander(platform, expanded=True):
            st.write(output)
            st.caption("Replayed demo output · no live model call")
    st.subheader("Multi-persona evaluation")
    st.bar_chart(DEMO_SCORES)
    st.subheader("Refinement history")
    st.markdown(f"**Initial draft**  \n{DEMO_REFINEMENT['initial']}")
    st.markdown(f"**Persona feedback**  \n{DEMO_REFINEMENT['feedback']}")
    st.markdown(f"**Accepted refinement**  \n{DEMO_REFINEMENT['accepted']}")
    st.caption("Example scores and token/cost values are simulated for the demo.")
