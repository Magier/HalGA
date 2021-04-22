import streamlit as st


def show():
    st.markdown("The termination criterion is a preset number of generations.")
    max_gens = st.slider("Number of Generations", min_value=1, max_value=100, value=50)
