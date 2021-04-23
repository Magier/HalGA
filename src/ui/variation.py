import copy
import inspect
from typing import List

import deap.tools
import numpy as np
import pandas as pd
import streamlit as st

import ga_library
import utils
from ui.individual_representation import color_adj_edge


def show_individual(mat: np.array, topology: List[str]):
    df_dag = pd.DataFrame(mat, index=topology, columns=topology)
    df_dag = df_dag.loc[topology][topology]
    st.markdown(f"Node order: `{topology}`")
    st.dataframe(df_dag.style.applymap(color_adj_edge), height=500)


def show_crossover():
    st.header("Crossover")
    st.markdown("""
    The crossover operator will just **recombine the topologies** of both parents without modifying the adjacency matrix.  
    The wrapper code turns individuals into a list of indices per individual, which is the format expected by DEAP.  
    _Note: DEAP modifies the individuals in place, but clones them before any variation is performed._
    """)

    st.image("assets/ordered_cx.png")

    st.subheader("The crossover operator in action")

    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    p1 = ga_library.Individual(*utils.generate_random_dag(nodes, p=.3))
    p2 = ga_library.Individual(*utils.generate_random_dag(nodes, p=.3))

    c1, c2 = ga_library.ordered_crossover(copy.deepcopy(p1), copy.deepcopy(p2))

    col_left, col_right = st.beta_columns(2)
    with col_left:
        st.subheader("Parent 1")
        show_individual(p1.mat, p1.topology)
        st.subheader("Parent 2")
        show_individual(p2.mat, p2.topology)

    with col_right:
        st.subheader("Child 1")
        show_individual(c1.mat, c1.topology)
        st.subheader("Child 2")
        show_individual(c2.mat, c2.topology)

    src_code = inspect.getsource(ga_library.ordered_crossover)
    st.code(src_code)

    with st.beta_expander("DEAP's Ordered Crossover code"):
        cx_code = inspect.getsource(deap.tools.cxOrdered)
        st.code(cx_code)


def show_mutation():
    st.header("Mutation")
    st.markdown("""
    The mutation operator performs a basic **bit flip** in the adjacency matrix.   
    Once an individual is chosen for mutation an **independent probability** is calculated for every value in the adjacency matrxi.  
    The mutation is constrained to the **strictly upper triangular** matrix in order to ensure that the result will be a DAG! 
    """)

    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    p = ga_library.Individual(*utils.generate_random_dag(nodes, p=.3))
    indpr = st.slider("Independent probability:", min_value=.0, max_value=1., value=.1)
    c, = ga_library.mutate_edge_flip(copy.deepcopy(p), indpr)

    col_left, col_right = st.beta_columns(2)
    with col_left:
        st.subheader("Parent")
        show_individual(p.mat, p.topology)
    with col_right:
        st.subheader("Mutant")
        show_individual(c.mat, c.topology)


    src_code = inspect.getsource(ga_library.mutate_edge_flip)
    st.code(src_code)


def show():
    show_crossover()
    show_mutation()
