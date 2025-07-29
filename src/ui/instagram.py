import streamlit as st

def instagram_tab():
    st.header("Instagram Content")
    final_state = getattr(st.session_state, "final_state", None)

    # Aggregated scores (show right below header)
    if hasattr(final_state, "aggregated_scores"):
        agg_scores = final_state.aggregated_scores.get("instagram", None)
    elif isinstance(final_state, dict) and "aggregated_scores" in final_state:
        agg_scores = final_state["aggregated_scores"].get("instagram", None)
    else:
        agg_scores = None

    if agg_scores is not None:
        st.markdown(f"**🎯Aggretated Score:** {agg_scores}")

    with st.container(height=200):
        if hasattr(final_state, "final_content"):
            content = final_state.final_content.get("instagram", {})
        elif isinstance(final_state, dict) and "final_content" in final_state:
            content = final_state["final_content"].get("instagram", {})
        else:
            content = {}
        
        post = content.get("post", "No Instagram content available.")
        hastags = content.get("hashtags", "No Instagram hastags available.")
        st.markdown(post)
        st.markdown(f"**{hastags}**")

    # Show refinement history for Instagram
    if hasattr(final_state, "refinement_history"):
        refinements = final_state.refinement_history.get("instagram", [])
    elif isinstance(final_state, dict) and "refinement_history" in final_state:
        refinements = final_state["refinement_history"].get("instagram", [])
    else:
        refinements = []

    if refinements and len(refinements) > 1:
        st.markdown("---")
        st.subheader("Refinement History")

        # Show initial draft explicitly
        st.markdown("**Initial Draft:**")
        st.markdown(refinements[0])

        for idx, refinement in enumerate(refinements[1:], start=1):
            with st.container():
                st.markdown(f"**Refinement {idx}:**")
                st.markdown(refinement["post"])
                st.markdown(refinement["hashtags"])
