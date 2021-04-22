import numpy as np
import pandas as pd
import pytest

from ga_library import Individual, mutate_edge_flip, crossover_k_points, ordered_crossover


@pytest.fixture
def sat_individual() -> Individual:
    cols = ["Difficulty", "Intelligence", "Grade", "SAT", "Letter"]
    adj_mat = np.matrix(
        "0 0 1 0 0; " "0 0 1 1 0;" "0 0 0 0 1;" "0 0 0 0 0;" "0 0 0 0 0"
    )
    return Individual(adj_mat, cols, fitness=-1337)


@pytest.fixture
def disconnected_individual() -> Individual:
    cols = ["A", "B", "C", "D", "E"]
    n = len(cols)
    adj_mat = np.zeros((n, n), np.int64)
    # df = pd.DataFrame(adj_mat, index=cols, columns=cols)
    return Individual(adj_mat, cols, fitness=1337)


@pytest.fixture
def fully_connected_individual() -> Individual:
    cols = ["A", "B", "C", "D", "E"]
    adj_mat = np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    return Individual(adj_mat, cols, fitness=1337)


class TestEdgeFlipMutation:
    def test_flip_all(self, disconnected_individual, fully_connected_individual):
        # Note: mutation returns tuple
        mutant, *_ = mutate_edge_flip(disconnected_individual, 1.0)
        expected_mat = fully_connected_individual.mat
        assert np.array_equal(expected_mat, mutant.mat)

    def test_flip_sat(self, sat_individual):
        # Note: mutation returns tuple
        mutant, *_ = mutate_edge_flip(sat_individual, 1.0)

        expected_mat = np.array([[0, 1, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        assert np.array_equal(expected_mat, mutant.mat)


class Test1PointCrossover:
    def test_empty_and_full_crossover(
        self, fully_connected_individual, disconnected_individual
    ):
        child1, child_2 = crossover_k_points(
            fully_connected_individual, disconnected_individual, k=1
        )

        assert False


class TestOrderedCrossover:
    def test_empty_and_full_crossover(
            self, fully_connected_individual, disconnected_individual
    ):
        child1, child_2 = ordered_crossover(
            fully_connected_individual, disconnected_individual, k=1
        )

        assert False
