import pydataset
import streamlit as st
import inspect
from ga_library import Individual

import ga_library
from ui.session_state import get_state
from ui import (
    description,
    framework_overview,
    individual_representation,
    fitness_evaluation,
    learn_structure,
    selection,
    termination_criterion,
    variation,
    deap_code
)

# TODO: add additional layouts to the streamlit-bd-cytoscape component like 'dagre' (see
#  https://github.com/QuantStack/ipycytoscape/blob/f461945a1e86de0687815eef3721e34734919d79/src/widget.ts and
#  https://github.com/QuantStack/ipycytoscape/blob/e697a11954f309c0e799963ed08984e6ba1eb53c/package.json)


PAGES = {
    "Description": description,
    "Framework": individual_representation,
    "Connecting the Dots": deap_code,
    "Learn!": learn_structure,
}

FRAMEWORK_PAGES = {
    "Overview": framework_overview,
    "Population": individual_representation,
    "Evaluation": fitness_evaluation,
    "Termination Criterion": termination_criterion,
    "Selection": selection,
    "Variation": variation,
}


def setup_sidebar() -> str:
    option = st.sidebar.radio("Outline", list(PAGES.keys()), index=0)

    if option == "Framework":
        option = st.sidebar.radio("Step:", list(FRAMEWORK_PAGES.keys()))
    else:
        fw_step = None

    return option


if __name__ == "__main__":
    st.set_page_config(page_title="HalGA", layout="wide", page_icon="🧬")
    page_name = setup_sidebar()

    state = get_state()
    if "toolbox" not in state:
        st.write("Initializing DEAP")
        data = pydataset.data("Caschool")
        state.toolbox = ga_library.setup_deap(ga_library.TOOLBOX, data)
    else:
        # st.write(f"found toolbox: {state.toolbox}")
        pass

    # if st.sidebar.button("Start"):
    #     learn_structure.show()
    # else:
    page = PAGES[page_name] if page_name in PAGES else FRAMEWORK_PAGES[page_name]
    st.title(page_name)
    page.show()
