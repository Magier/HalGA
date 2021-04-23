from typing import Any, Dict
import datetime
import streamlit as st
import pickle
import pandas as pd
import altair as alt
from ga_library import Individual
import networkx
from ui.individual_representation import visualize_individual

from ui.session_state import get_state


def show_summary():
    st.header("Summary")
    state = get_state()
    if "edge_prob" in state:
        st.write(f"Edge Probability: {state.edge_prob}")
    pass


def load_results() -> Dict[str, Any]:
    with open("run_res.pkl", "rb") as f:
        res = pickle.load(f)
    return res


def create_fitness_evolution_chart(df: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(df.rename_axis("gen").reset_index()).mark_line().encode(
        x=alt.X("gen:N", title="Generation"),
        y=alt.Y("max:Q", title="Fitness", scale=alt.Scale(domain=(0, 1.))),
        tooltip=["gen", "max", "nevals"]
    )
    return chart


def show_results():
    res = load_results()

    logbook = res["logbook"]

    df_log = pd.DataFrame(logbook).drop(["gen"], axis=1)
    chart = create_fitness_evolution_chart(df_log)
    st.altair_chart(chart)

    duration = datetime.timedelta(seconds=round(res["duration"]))
    st.markdown(f"The run took **{duration} hours**")

    best_per_gen = res["best_per_gen"]
    gen = st.slider("Generation:", min_value=0, max_value=len(best_per_gen)-1)
    st.subheader(f"Best individual of generation {gen}")
    best_ind = best_per_gen[gen][0]
    st.write(f"Fitness: {best_ind.fitness:.2f}")

    visualize_individual(best_ind.mat, best_ind.topology)

    # history = res["history"]
    # graph = networkx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()  # Make the graph top-down
    # colors = [evaluate(history.genealogy_history[i])[0] for i in graph]

    # fig, ax = plt.subplots()
    # p= networkx.draw(graph)
    # st.pyplot(fig)


def show():
    hp = st.sidebar.beta_expander(
        "Hyperparameters",
    )
    pop_size = hp.number_input("Population Size", min_value=2, max_value=500, value=50)
    num_elites = hp.number_input("Number of Elites", min_value=0, max_value=10, value=1)
    cx_pr = hp.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=1.)
    mut_pr = hp.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2)

    show_summary()

    if st.button("Start Search"):
        st.balloons()
        st.experimental_rerun()
    # show_results()
