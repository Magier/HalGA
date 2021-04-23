import streamlit as st


def show_checklist():
    st.header("Checklist")
    st.checkbox("Define individual representation")
    st.checkbox("Randomly generate 1st generation")
    st.checkbox("Implement fitness evaluation")
    st.checkbox("Decide on termination criterion(s)")
    st.checkbox("Choose method for parent selection")
    st.checkbox("Choose method  for replacement")
    st.checkbox("Implement crossover operator")
    st.checkbox("Implement mutation operator")
    st.checkbox("Configure hyperparameters")


def show():
    # overview flowchart
    st.image("assets/EvoluationaryAlgorithm_base.png")
    
    show_checklist()
