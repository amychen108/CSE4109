import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        .stApp {
            max-width: 800px;
            margin: 0 auto;
        }
        .stButton button {
            background-color: #2c7be5;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #1a5fbd;
        }
        .stAlert {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)