import numpy as np
import pandas as pd
import networkx

from src import utils


class TestDependencyListToAdjacencyMatrixConversion:
    def test_basic_conversion(self):
        deps = [
            ["Difficulty", ["Grade"]],
            ["Intelligence", ["Grade", "SAT"]],
            ["Grade", ["Letter"]],
        ]

        cols = ["Difficulty", "Intelligence", "Grade", "SAT", "Letter"]
        df_res = utils.dependencies_to_adjacency_matrix(deps, cols)

        adj_mat = np.matrix("0 0 1 0 0; "
                            "0 0 1 1 0;"
                            "0 0 0 0 1;"
                            "0 0 0 0 0;"
                            "0 0 0 0 0")

        df_exp = pd.DataFrame(adj_mat, index=cols, columns=cols)
        pd.testing.assert_frame_equal(df_exp, df_res)


class TestAdjacencyMatrixToDependencyListConversion:
    def test_basic_conversion(self):
        cols = ["Difficulty", "Intelligence", "Grade", "SAT", "Letter"]
        adj_mat = np.matrix("0 0 1 0 0; "
                            "0 0 1 1 0;"
                            "0 0 0 0 1;"
                            "0 0 0 0 0;"
                            "0 0 0 0 0")

        deps = utils.adjacency_matrix_to_deps(adj_mat, cols)
        assert deps == [
            ("Difficulty", "Grade"),
            ("Intelligence", "Grade"),
            ("Intelligence", "SAT"),
            ("Grade", "Letter"),
        ]


class TestGrapnGeneration:
    def test_generate_small_dag(self):
        nodes = ["Bob", "Eve", "Alice", "Mallory"]
        mat, topo = utils.generate_random_dag(nodes, .5)
        n = len(nodes)
        assert mat.shape == (n, n)
