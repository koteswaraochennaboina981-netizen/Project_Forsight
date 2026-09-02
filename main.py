import streamlit as st
import runpy

st.set_page_config(
    page_title="Project Foresight",
    layout="wide"
)

runpy.run_path("ml_part/08_dashboard.py", run_name="__main__")
