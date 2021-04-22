# HalGA
An educational tool showing how an evolutionary algorithm can be use to learn the structure of a Bayesian network.

The approach is heavily inspired by the example provided in the documentation of [Halerium](https://hal.erium.io/examples/04_causal_structure/03-causal_structure_calschool.html), where the causal structure is defined for the [California School](https://rdrr.io/cran/Ecdat/man/Caschool.html) data set.

### Core Idea:
Learn a Bayesian Network purely from observational data. The resulting model should do well on predicting `mathscr`, `readscr` and `testscr`.  
This task can be broken down into the subtasks of:
- **learning the causal structure** → use a **Genetic Algorithm**
- learning the parameters → train Bayesian Network on data using Halerium for proposed proposed structure


## Dataset

#### Description   
a cross-section from 1998-1999  
number of observations: 420  
observation: schools  
country: United States

#### Columns
- `distcod`: district code
- `county`: county
- `district`: district
- `grspan`: grade span of district
- `enrltot`: total enrollment 
- `teachers`: number of teachers
- `calwpct`: percent qualifying for CalWORKS
- `mealpct`: percent qualifying for reduced-price lunch
- `computer`: number of computers
- `testscr`: average test score (read.scr+math.scr)/2
- `compstu`: computer per student
- `expnstu`: expenditure per student
- `str`: student teacher ratio
- `avginc`: district average income
- `elpct`: percent of English learners
- `readscr`: average reading score
- `mathscr`: average math score

#### Source
California Department of Education https://www.cde.ca.gov. 


## Used Tools
- [DEAP](https://github.com/deap/deap) as framework for genetic algorithms
- [HALerium](https://hal.erium.io/) to evaluate fitness of causal structure


## Note
This is an _educational tool_ intended as showcase how a **Genetic Algorithm** can be applied to a real problem.  
It does **not** try to challenge the state-of-the-art nor does it use any novel operations.
