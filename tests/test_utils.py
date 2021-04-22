import numpy as np
import pandas as pd
import networkx

from src import utils


class TestAdjacencyListToDependenciesConversion:
    def test_basic_conversion(self):
        cols = ["Difficulty", "Intelligence", "Grade", "SAT", "Letter"]
        col_map = dict(enumerate(cols))
        adj_list = [
            [2],
            [2, 3],
            [4],
            [],
            []
        ]
        deps = utils.adjacency_list_to_deps(adj_list, col_map)
        assert deps == [
            ["Difficulty", ["Grade"]],
            ["Intelligence", ["Grade", "SAT"]],
            ["Grade", ["Letter"]],
        ]

    def test_structured_conversion(self):
        adj_list = [
            [1, 6, 8, 4],
            [8],
            [11, 12],
            [11, 12],
            [6],
            [],
            [11, 12],
            [1, 4],
            [11, 12],
            [2, 3],
            [2, 3, 11, 12],
            [5],
            [5]
        ]
        deps = utils.adjacency_list_to_deps(adj_list, cols)
        assert deps == [
            ["str", ["mathscr", "readscr"]],
            [["teachers", "enrltot"], "str"],
            [["expnstu", "enrltot"], "teachers"],
            ["compstu", ["mathscr", "readscr"]],
            [["computer", "enrltot"], "compstu"],
            [["expnstu", "enrltot"], "computer"],
            [["mealpct", "calwpct"], ["mathscr", "readscr"]],
            ["avginc", ["mealpct", "calwpct"]],
            ["elpct", ["mathscr", "readscr", "mealpct", "calwpct"]],
            ["readscr", "testscr"],
            ["mathscr", "testscr"],
        ]


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

        df_mat = pd.DataFrame(adj_mat, index=cols, columns=cols)

        deps = utils.adjacency_matrix_to_deps(df_mat)
        assert deps == [
            ["Difficulty", ["Grade"]],
            ["Intelligence", ["Grade", "SAT"]],
            ["Grade", ["Letter"]],
        ]


class TestGrapnGeneration:
    def test_generate_small_dag(self):
        nodes = ["Bob", "Eve", "Alice", "Mallory"]
        df = utils.generate_random_dag(nodes, .5)
        df.index = nodes
        df.columns = nodes
        n = len(nodes)
        assert df.shape == (n, n)