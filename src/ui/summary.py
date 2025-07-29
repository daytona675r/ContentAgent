import textwrap
import streamlit as st

from agent.social_content_graph import ContentState

def summary_tab():
    st.subheader("Content Summary")
    final_state = getattr(st.session_state, "final_state", None)

    # Check if final_content exists and is not empty
    if hasattr(final_state, "final_content"):
        final_content = final_state.final_content
    elif isinstance(final_state, dict) and "final_content" in final_state:
        final_content = final_state["final_content"]
    else:
        final_content = None

    if not final_content:
        st.info("No content generated yet.")
        return

    # Create a card per platform
    for platform, content in final_content.items():
        # For compatibility: content may be a string or dict
        if isinstance(content, dict):
            post_text = content.get("post", content.get("text", "No post available"))
        else:
            post_text = content or "No post available"

        # Get score
        if hasattr(final_state, "aggregated_scores"):
            avg_score = final_state.aggregated_scores.get(platform, 0)
        elif isinstance(final_state, dict) and "aggregated_scores" in final_state:
            avg_score = final_state["aggregated_scores"].get(platform, 0)
        else:
            avg_score = 0

        # Shorten post for preview
        preview = textwrap.shorten(post_text, width=150, placeholder="...")

        # Choose color for score badge
        score_color = "green" if avg_score >= 8 else "orange" if avg_score >= 5 else "red"

        # Card UI
        with st.container():
            st.markdown(f"### {platform.capitalize()}")
            st.markdown(f"**Score:** :{score_color}[{avg_score}/10]")
            st.markdown(f"> {preview}")
