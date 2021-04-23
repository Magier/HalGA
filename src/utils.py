import random
from typing import Dict, List, Optional, Tuple, Union

import networkx
import networkx as nx
import numpy as np
import pandas as pd
from ipycytoscape import CytoscapeWidget
import halerium.core as hal


def show_hal_graph(g: hal.Graph) -> CytoscapeWidget:
    deps = dependencies_from_hal_graph(g)
    return show_graph(deps)


def show_graph(
        dependencies: List[Tuple[str, str]]
) -> Union[Tuple[Dict, List], CytoscapeWidget]:
    elements, style, layout = get_cytoscape_params(dependencies, True)

    cytoscapeobj = CytoscapeWidget()
    cytoscapeobj.graph.add_graph_from_json(elements)

    cytoscapeobj.set_style(style)

    cytoscapeobj.cytoscape_layout = layout

    return cytoscapeobj


def get_cytoscape_params(
        edgelist: List[Tuple], nodes: Optional[List[str]] = None, for_jupyter: bool = False
):
    edges = [
        {"data": {"id": f"{s}->{t}", "source": s, "target": t}} for s, t in edgelist
    ]

    node_iter = set(sum(edgelist, ())) if nodes is None else nodes
    nodes = [{"data": {"id": n}} for n in node_iter]
    style = [
        {"selector": "core", "style": {"background": "blue"}},
        {
            "selector": "node",
            "style": {
                "label": "data(id)",
                "background-color": "gray",
                "font-size": "2em",
                "text-valign": "center",
                "text-halign": "center",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "width": 5,
                # "line-color": "#ddd",
                "background-fill": "linear-gradient(yellow, darkgray)",
                "target-arrow-color": "darkgray",
            },
        },
    ]

    layout = {"name": "breadthfirst", "directed": True, "circle": True}
    if for_jupyter:
        elements = {"nodes": nodes, "edges": edges}
    else:
        elements = nodes + edges
    return elements, style, layout


def get_node_name(variable_name: str) -> str:
    return variable_name.replace("graph/var_", "")


def is_regular_variable(variable_name: str) -> bool:
    return variable_name.startswith("graph/var_")


def dependencies_from_hal_graph(g: hal.Graph) -> List:
    g_viz = hal.gui.GraphVisualizer(g)
    g_dict = dict(g_viz._get_json())
    edges = []
    for edge in g_dict["variable_dependencies"]["graph"]:
        src = edge["source"]
        dst = edge["target"]
        if is_regular_variable(src) and is_regular_variable(dst):
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)
            edges.append((src_name, dst_name))
    return edges


def dependencies_to_adjacency_matrix(
        deps: List[List], columns: List[str]
) -> pd.DataFrame:
    """

    :param deps:
    :param columns:
    :return:
    """
    rows = {c: {c: 0} for c in columns}
    # nodes = set()

    for src, targets in deps:
        assert isinstance(src, str), "multiple sources are currently not supported"
        if not isinstance(targets, list):
            targets = [targets]
        # nodes.update([src] + targets)  # keep track of all observed nodes to obtain full adj matrix
        target_cols = rows.get(src, {})
        new_cols = {c: 1 for c in targets}
        rows[src] = {**target_cols, **new_cols}

    # add rows for nodes without any outgoing arrows
    # if columns is None:
    #     columns = list(rows.keys())

    # for name in set(columns) - set(rows.keys()):
    #     rows[name] = {
    #         name: 0
    #     }  # empty entries would be dropped at initialization -> in valid DAG no node points to
    #     # itself
    #

    df_mat = pd.DataFrame.from_dict(rows, orient="index", columns=columns).fillna(0)

    # make sure both dimensions have the same order, as expected from a adj. matrix

    return df_mat.astype(int)


def adjacency_matrix_to_deps(mat: np.array, topo: List[str], keep_empty: bool = False) -> List:
    """
    Convert an adjacency matrix to a list of dependencies.
    A dependency itself is a list of length 2, where the first element is the source node and
    the second element is a list of destination nodes.
    :param mat: the adjacency matrix as a 2D numpy array
    :param topo: a list of nodes sorted in topological order
    :param keep_empty: a flag specifying whether to add nodes with no outgoing edges as well
    :return: a list of dependencies
    """
    edges = [(topo[r], topo[c]) for r, c in np.argwhere(mat > 0)]

    if keep_empty:
        used_nodes = set([src for src, targets in edges])
        edges += [(node, []) for node in topo if node if node not in used_nodes]

    return edges


def adjacency_matrix_to_deps2(df_mat: pd.DataFrame, row_per_edge: bool = True, keep_empty: bool = False) -> List:
    """
    Convert an adjacency matrix to a list of dependencies.
    A dependency itself is a list of length 2, where the first element is the source node and
    the second element is a list of destination nodes.
    :param df_mat:  the adjacency matrix as pandas DataFrame
    :return: a list of dependencies
    """

    dep_list = [
        [src, s_adj[s_adj > 0].index.tolist()]
        for src, s_adj in df_mat.iterrows()
        if keep_empty or sum(s_adj) > 0
    ]

    if not row_per_edge:
        return dep_list

    edgelist = [(src, target) for src, targets in dep_list for target in targets]
    return edgelist


# def generate_random_dag(
#     nodes: List, p: float, as_dataframe: bool = True
# ) -> Union[np.ndarray, pd.DataFrame]:
def generate_random_dag(nodes: List, p: float) -> Tuple[np.ndarray, List[str]]:
    # approach found on stack overflow: https://stackoverflow.com/questions/13543069/how-to-create-random-single-source-random-acyclic-directed-graphs-with-negative
    n = len(nodes)
    random_graph = networkx.fast_gnp_random_graph(n, p, directed=True)
    random_dag = networkx.DiGraph([(i, j) for (i, j) in random_graph.edges() if i < j])

    # due to the filtering nodes can get lost -> ensure resulting DAG has the desired number of nodes
    if random_dag.number_of_nodes() < n:
        random_dag.add_nodes_from(random_graph.nodes)

    # validate resulting DAG. An invalid DAG means there is a major bug in the creation
    if not networkx.is_directed_acyclic_graph(random_dag):
        raise ValueError("Created graph is not a valid DAG")

    # assign random labels to generated nodes in DAG
    rand_nodes = nodes[:]  # create shallow copy to avoid altering input argument
    random.shuffle(rand_nodes)
    node_mapping = {i: n for i, n in enumerate(rand_nodes)}

    # sorted_nodes = sorted(nodes)
    random_dag = networkx.relabel_nodes(random_dag, node_mapping)

    df = networkx.convert_matrix.to_pandas_adjacency(random_dag, nodes).astype(int)
    topology = list(nx.topological_sort(random_dag))
    df_sorted = df.loc[topology][topology]
    return df_sorted.values, topology
    # return (
    #     networkx.convert_matrix.to_numpy_array(random_dag, nodes).astype(int),
    #     rand_nodes,
    # )
