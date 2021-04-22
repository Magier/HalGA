import copy
import pickle
import time
from dataclasses import dataclass
from functools import partial
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import deap.base
import numpy as np
import pandas as pd
import pydataset
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import halerium
from halerium import CausalStructure, Evaluator
from networkx import topological_sort
import networkx as nx

import utils
from utils import generate_random_dag

MIN_VAL = np.finfo(np.float64).min
TOOLBOX = base.Toolbox()


@dataclass
class Individual:
    mat: np.array  # adjacency matrix sorted topologically
    topology: list  # node names in topological order
    fitness: float = np.nan  # fitness score associated with the individual

    def __len__(self):
        return self.mat.shape[0]

    def __eq__(self, other):
        return np.array_equal(self.mat, other.mat) and self.topology == other.topology


def prepare_eval_data(df: pd.DataFrame) -> Dict[str, Any]:
    np.random.seed(123)
    random_indices = np.random.choice([True, False], size=len(df), p=[0.75, 0.25])
    df_train = df.iloc[random_indices]
    df_test = df.iloc[~random_indices]
    outputs = {"mathscr", "readscr", "testscr"}
    inputs = set(df_train.columns[4:]) - outputs
    return {
        "df_train": df_train,
        "df_test": df_test,
        "inputs": inputs,
        "outputs": outputs,
    }


def setup_deap(toolbox: base.Toolbox, df: pd.DataFrame, edge_prob: float = 0.25):
    nodes = list(df.columns[4:])
    eval_kwargs = prepare_eval_data(df)

    # Alternatively, one can define the types for the individual using DEAP's creator.
    # creator.create("FitnessMax", base.Fitness, weights=(0.0, 0.0, 0.0))
    # creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox.register("generate_dag", generate_random_dag, nodes, p=edge_prob)
    toolbox.register(
        "individual",
        lambda init_fn: Individual(*init_fn(), fitness=np.nan),
        toolbox.generate_dag,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", ordered_crossover)
    toolbox.register("mutate", mutate_edge_flip, indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_with_halerium, **eval_kwargs)


# def evaluate(
#     individual: Individual, data: pd.DataFrame
# ):  # , inputs: Iterable, outputs: Iterable):
#     deps = utils.adjacency_matrix_to_deps(individual.mat, row_per_edge=True)
#     model = BayesianModel(deps)
#     # k2 = K2Score(data)
#     # bic = BicScore(data)
#     # k2_score = k2.score(model)
#     # bic_score = bic.score(model)
#
#     bic_score = 42
#     return (bic_score,)  # , k2_score


def mutate_edge_flip(ind: Individual, indpb: float) -> Tuple[Individual]:
    """Mutates the individual by flipping edges in the adjacency matrix.
    The probability of an edge being flipped is independently applied to all edges in
     the upper triangular matrix, so it remains a valid DAG.
    :param ind: individual to be mutated.
    :param indpb: independent probability for each attribute to be exchanged to another position.
    :returns: a tuple of one individual
    """
    triang_indices = np.triu_indices_from(ind.mat, k=1)
    for row, col in zip(*triang_indices):
        if random.random() < indpb:
            ind.mat[(row, col)] = int(not ind.mat[(row, col)])
    return (ind,)


def ordered_crossover(
    ind1: Individual, ind2: Individual
) -> Tuple[Individual, Individual]:
    """
    Mate two individuals by recombining their respective topology order. This operator
    produces two offsprings, where each inherits the unchanged adjacency matrix of a parent.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two offsprings.
    """
    if ind1 == ind2:
        return ind1, ind2

    # turn list of topologies into a list of indices
    cols = sorted(ind1.topology)
    idx_map = {n: i for i, n in enumerate(cols)}
    p1_nodes_order = [idx_map[n] for n in ind1.topology]
    p2_nodes_order = [idx_map[n] for n in ind2.topology]

    # for actual crossover operation DEAPs ordered crossover function is used
    ch1_node_order, ch2_node_order = tools.cxOrdered(p1_nodes_order, p2_nodes_order)

    # update the topology list on the resulting offsprings
    ind1.topology = [cols[i] for i in ch1_node_order]
    ind2.topology = [cols[i] for i in ch2_node_order]

    return ind1, ind2


def evaluate_with_halerium(
    individual,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    inputs: List[str],
    outputs: Set[str],
):
    # turn adjacency matrix and the topology into a list of dependencies
    deps = utils.adjacency_matrix_to_deps(
        individual.mat, individual.topology, keep_empty=True
    )
    try:
        structure = CausalStructure(deps)
        structure.train(df_train, method="MGVI")
        performance = structure.evaluate_objective(
            Evaluator, data=df_test, inputs=inputs, outputs=outputs, metric="r2"
        )
        math_score = performance["mathscr"]
        read_score = performance["readscr"]
        test_score = performance["testscr"]
    except halerium.causal_structure.dependencies.CyclicDependencyError as exc:
        math_score, read_score, test_score = MIN_VAL, MIN_VAL, MIN_VAL

    # return a single score which is is the weighted sum of individual scores
    return 0.25 * math_score + 0.25 * read_score + 0.5 * test_score


def print_best(ind: Individual, gen: int) -> None:
    print(f"=========== Best in generation {gen}: ============")
    print(f"Fitness: {ind.fitness:.3f}")
    print(f"Topology: {ind.topology}")
    print(f"Mat: {ind.mat}")


def get_fittest_individuals(population: List[Individual], n: int = 1) -> List[Individual]:
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    fittest_individuals = sorted_pop[:n]
    return fittest_individuals


def log_generation_stats(
    gen: int,
    population: List[Individual],
    n_evals: int,
    stats: deap.tools.Statistics,
    log: deap.tools.Logbook,
    best_per_gen: List,
) -> None:
    fittest_ind = get_fittest_individuals(population, 1)
    print_best(fittest_ind[0], gen)
    best_per_gen.append(fittest_ind)
    # Append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    log.record(gen=gen, nevals=n_evals, **record)


def initialize_instrumentation() -> Dict[str, Any]:
    stats = tools.Statistics(lambda ind: ind.fitness)
    stats.register("max", np.max)
    log = tools.Logbook()
    log.header = ["gen", "nevals"] + (stats.fields if stats else [])

    return {"best_per_gen": [], "stats": stats, "log": log}


def learn_causal_structure(
    toolbox: base.Toolbox,
    pop_size: int = 10,
    crossover_pr: float = 1,
    mutation_pr: float = 0.2,
    num_elites: int = 1,
    max_gens: int = 50,
):
    """
    Perform the structur learning task using a genetic algorithm
    :param toolbox: registry of tools provided by DEAP
    :param pop_size: the number of individuals per generation
    :param crossover_pr: the crossover rate for every (monogamous) couple
    :param mutation_pr: the mutation rate for every individual
    :param num_elites:
    :param max_gens: the maximum number of generations
    :return:
    """
    # initialize a collection of instrumentation utilities to facilitate later analysis
    instrumentation = initialize_instrumentation()

    # ====== 0️⃣ initialize population  ======
    population = toolbox.population(n=pop_size)

    # ====== 1️⃣ Evaluate the entire population ======
    n_evals = evaluate_population(population, toolbox)
    # Log initial stats for later analysis
    log_generation_stats(0, population, n_evals, **instrumentation)

    # ====== 2️⃣ the loop is the only termination criterion ======
    for gen in range(max_gens):
        elites = get_fittest_individuals(population, num_elites)

        # ====== 3️⃣ Parent selection ======
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # ====== 4️⃣ Apply crossover and mutation on the offspring ======
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # crossover probability applies to every couple
            if random.random() < crossover_pr:
                toolbox.mate(child1, child2)
                child1.fitness = np.nan
                child2.fitness = np.nan

        # mutation probability applies to every individual
        for mutant in offspring:
            if random.random() < mutation_pr:
                toolbox.mutate(mutant)
                mutant.fitness = np.nan

        # ====== 5️⃣ Evaluate the individuals with an invalid fitness ======
        n_evals = evaluate_population(offspring, toolbox)
        # Log intermediary stats for later analysis
        log_generation_stats(gen+1, population, n_evals, **instrumentation)

        # ====== 6️⃣ Replacement ======
        # The population is entirely replaced by the offspring, except for the top elites
        fittest_offsprings = get_fittest_individuals(offspring, pop_size - num_elites)
        population[:] = elites + fittest_offsprings

    # ====== 7️⃣ Return final population ======
    return population, instrumentation


def evaluate_population(offspring, toolbox) -> int:
    invalid_ind = [ind for ind in offspring if np.isnan(ind.fitness)]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness = fit
    return len(invalid_ind)


if __name__ == "__main__":
    df = pydataset.data("Caschool")
    setup_deap(TOOLBOX, df, edge_prob=0.1)

    tic = time.time()

    history = tools.History()
    TOOLBOX.decorate("mate", history.decorator)
    TOOLBOX.decorate("mate", history.decorator)

    pop, instrumentation = learn_causal_structure(TOOLBOX)
    toc = time.time()

    print(f"Execution took {(toc-tic):.2f} seconds")
    logbook = instrumentation["log"]

    with open("run_res.pkl", "wb") as f:
        results = {
            "duration": toc-tic,
            "population": pop,
            "history": history,
            "logbook": logbook,
            "best_per_gen": instrumentation["best_per_gen"],
        }
        pickle.dump(results, f)
        print("Saved results!")

    print(logbook)
