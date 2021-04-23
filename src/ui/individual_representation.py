from typing import List, Tuple

import networkx
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_bd_cytoscapejs import st_bd_cytoscape

import ga_library
import utils
import inspect

from ui.session_state import get_state


def color_adj_edge(val):
    color = 'lightgray' if val == 0 else 'black; font-weight: bold'
    return f"color: {color}"


def show_individual_internals(mat: np.array, topology: List):
    st.subheader("Topological order")
    st.markdown(f"Node order: `{topology}`")
    st.subheader("Adjacency Matrix")
    show_topo_order = st.checkbox("Show in topological order")
    df_dag = pd.DataFrame(mat, index=topology, columns=topology)
    if not show_topo_order:
        sorted_cols = topology if show_topo_order else sorted(topology)
        df_dag = df_dag.loc[sorted_cols][sorted_cols]
    st.dataframe(df_dag.style.applymap(color_adj_edge), height=500)


def show_dag(mat: np.array, topology: List):
    st.subheader("Graph Representation")
    deps = utils.adjacency_matrix_to_deps(mat, topology)
    elements, style, layout = utils.get_cytoscape_params(deps, nodes=topology)
    st_bd_cytoscape(elements, style, layout, key="dag_gen_example")


def visualize_individual(mat: np.array, topology: List, show_graph: bool = True, show_mat: bool = True) -> None:
    if show_mat and show_graph:
        col_left, col_mid, col_right = st.beta_columns([0.475, 0.05, 0.475])
        with col_left:
            show_dag(mat, topology)
        with col_right:
            show_individual_internals(mat, topology)
    elif show_graph:
        show_dag(mat, topology)
    elif show_mat:
        show_individual_internals(mat, topology)


def show_individual():
    nodes = [
        "avginc",
        "calwpct",
        "compstu",
        "computer",
        "elpct",
        "enrltot",
        "expnstu",
        "mathscr",
        "mealpct",
        "readscr",
        "str",
        "teachers",
        "testscr",
    ]
    state = get_state()

    st.subheader("Individual Representation")
    ind_class = inspect.getsource(ga_library.Individual)
    st.code(ind_class)

    st.subheader("Example")
    # curr_edge_prob = state.get("edge_prob", .5)
    state.edge_prob = st.slider("Edge Probability", min_value=0., max_value=1., value=.5)
    mat, topology = utils.generate_random_dag(nodes, p=state.edge_prob)
    visualize_individual(mat, topology)


def show_individual_generation_src():
    with st.beta_expander("Code to generate random DAG"):
        src_code = inspect.getsource(utils.generate_random_dag)
        st.code(src_code)


def show():
    st.header("Individual")
    show_individual()
    show_individual_generation_src()
