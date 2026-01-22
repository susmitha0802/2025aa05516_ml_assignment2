import streamlit as st

def style():

    st.markdown(
        """
        <style>
            .stSidebar {
                width:441px !important;
            }

            .stMainBlockContainer {
                padding: 3rem;
            }

            .stHeading {
                text-align: center;
            }

            .st-key-upload_file p, .st-key-select_model p {
                font-size: medium;
            }

        </style>
        """,
        unsafe_allow_html=True
    )