import streamlit as st
from agent.social_content_graph import social_content_graph
from agent.social_content_graph import ContentState
from tools.business_info import get_business_info
from tools.insights import get_market_insights

def run_content_generation(topic:str):
    # Initialize state with topic + insights
    insights = " ".join(get_market_insights())
    business_info = " ".join(get_business_info())
    initial_state = ContentState(
        topic=topic,
        summarized_insights=insights,
        business_info=business_info,
        final_content={}, 
        current_node=None
    )

    with st.status("Generating content...", expanded=True) as status:
        final_state = None
        for event in social_content_graph.stream(initial_state):
            node_name = next(iter(event.keys()))
            state_dict = event[node_name]
            node = state_dict.get("current_node")
            # Update UI based on current_node
            if node == "generate_content_node":
                status.update(label="Generating drafts for all platforms...", state="running")
            elif node == "score_personas_node":
                status.update(label="Scoring drafts with personas...", state="running")
            elif node == "refine_content_node":
                status.update(label="Refining low-scoring drafts...", state="running")

            final_state = state_dict

        # Mark done
        status.update(label="Content created successfully!", state="complete")
    return final_state