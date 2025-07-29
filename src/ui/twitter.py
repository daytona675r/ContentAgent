import streamlit as st

def twitter_tab():
    st.header("Twitter Content")
    final_state = getattr(st.session_state, "final_state", None)

    # Aggregated scores (show right below header)
    if hasattr(final_state, "aggregated_scores"):
        agg_scores = final_state.aggregated_scores.get("twitter", None)
    elif isinstance(final_state, dict) and "aggregated_scores" in final_state:
        agg_scores = final_state["aggregated_scores"].get("twitter", None)
    else:
        agg_scores = None

    if agg_scores is not None:
        st.markdown(f"**🎯Aggretated Score:** {agg_scores}")
        
    with st.container(height=200):
        if hasattr(final_state, "final_content"):
            twitter_content = final_state.final_content.get("twitter", {})
        elif isinstance(final_state, dict) and "final_content" in final_state:
            twitter_content = final_state["final_content"].get("twitter", {})
        else:
            twitter_content = {}
        
        twitter_post = twitter_content.get("post", "No Twitter content available.")
        twitter_hastags = twitter_content.get("hashtags", "No Twitter hastags available.")
        st.markdown(twitter_post)
        st.markdown(f"**{twitter_hastags}**")

    # Show refinement history 
    if hasattr(final_state, "refinement_history"):
        refinements = final_state.refinement_history.get("twitter", [])
    elif isinstance(final_state, dict) and "refinement_history" in final_state:
        refinements = final_state["refinement_history"].get("twitter", [])
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
        


