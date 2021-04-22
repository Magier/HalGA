import inspect

import streamlit as st

import ga_library


def show_setup():
    st.header("Setup of toolbox and its operators")
    st.markdown("""
    DEAP provides a set of tools and utility function out of the box.   
    The most important one is the `toolbox`, which serves as a registry of all available operations and functions.  
    New operations and functions can be added by calling the `register` function, which has the following arguments
    - the name under which the registered function can be accessed, i.e. `toolbox.<name>
    - the reference actual function (without invoking the function)
    - any arguments and keyword arguments, that will be forwarded to the function
    The registered function is a `partial` function, which has the provided arguments set but can be invoked at a later point.
    
    """)
    setup_src = inspect.getsource(ga_library.setup_deap)
    st.code(setup_src)


def show_algorithm():
    st.header("Genetic Algorithm")
    ga_src = inspect.getsource(ga_library.learn_causal_structure)
    st.code(ga_src)


def show():
    show_setup()
    show_algorithm()
