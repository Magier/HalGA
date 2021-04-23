import inspect

import deap.tools
import streamlit as st


def show_parent_selection():
    st.header("Parent Selection")
    st.markdown("Parents are selected using a **tournament selection**.")
    st.image("assets/tournament_selection.jpg")

    st.markdown("The DEAP framework supports this type of Selection out of the box:")
    with st.beta_expander("DEAP's tournament selection code"):
        src_code = inspect.getsource(deap.tools.selTournament)
        st.code(src_code)


def show_replacement():
    st.header("Replacement")
    st.markdown("""
    The replacement strategy is a full **generation replacement**, but a predefined number of **elites** are carried over.  
    This means all parents (except the top elites) die, at the end of a generation and
    the created children will replace them.   
    To avoid any premature convergence it important to keep the population at the same size.
    """)
    st.code("""
    # The population is entirely replaced by the offspring, except for the top elites
    elites = get_fittest_individuals(population, num_elites)
    # ...
    # perform selection and variation steps
    # ...
    fittest_offsprings = get_fittest_individuals(offspring, pop_size - num_elites)
    population[:] = elites + fittest_offsprings
    """)


def show():
    show_parent_selection()
    show_replacement()


