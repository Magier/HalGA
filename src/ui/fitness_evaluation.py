import inspect

import streamlit as st

import ga_library


def show_code():
    with st.beta_expander("Code to evaluate a proposed DAG"):
        src_code = inspect.getsource(ga_library.evaluate_with_halerium)
        st.code(src_code)


def show():
    st.markdown("""
    For the evaluation of the fitness the parameters for a Bayesian Network with the evaluated structure are learned using Halerium.
    
    As a fitness value a single weighted score comprised of the `readscr`, `mathscr` and `testscr` are returned.
    The weights are a subjective choice and can be freely adjusted.
    
    In case of an error the minimum float value is used for each score, which means the individual will be sorted out by the algorithm. 
    """)
    show_code()

