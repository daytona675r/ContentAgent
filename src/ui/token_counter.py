import streamlit as st

def floating_token_box(final_state):
    if final_state:
        st.markdown(
            """
            <style>
            .floating-token-box {
                position: fixed;
                bottom: 32px;
                right: 32px;
                width: 10vw;
                min-width: 180px;
                max-width: 320px;
                z-index: 9999;
                background: rgba(255,255,255,0.85);
                border-radius: 12px;
                box-shadow: 0 2px 16px rgba(0,0,0,0.12);
                padding: 1.2em 1em 1.2em 1em;
                font-size: 1.05em;
                color: #222;
                border: 1px solid #e0e0e0;
            }
            .floating-token-box div { margin-bottom: 0.5em; }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="floating-token-box">'
            f'<b>Usage & Cost</b><br>'
            f'Tokens used: <b>{final_state["token_count"]}</b><br>'
            f'Est. price: <b>${final_state["price_usd"]:.4f}</b>'
            f'</div>',
            unsafe_allow_html=True
        )
