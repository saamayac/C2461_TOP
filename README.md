# Team Orienteering Problem (TOP) Solver

This repository contains various implementations for solving the Team Orienteering Problem (TOP) using different algorithms and methods. The repository is organized into several modules, each focusing on specific aspects of the TOP.

## Repository Structure

**Base.py**: This module contains basic methods and utilities that are essential for all algorithm implementations in this repository. These foundational methods are shared across different algorithms, providing a consistent base for the development and execution of various solution strategies.

**Evolutionary.py**: This module implements an evolutionary algorithm that is enhanced by Variable Neighborhood Descent (VND). The combination of evolutionary algorithms and VND leverages the global search capability of evolutionary strategies and the local search strength of VND, making it an effective approach for solving the TOP.

**InitSol.py**: This module focuses on generating initial solutions for the TOP. It includes one constructive algorithm that uses heuristic methods to build an initial solution and two random algorithms that provide diverse starting points. These initial solutions serve as the foundation for further optimization through local search and metaheuristic techniques.

**LocalSearch.py**: This module implements five local search algorithms designed to refine and improve solutions. The strategies include Swap (First Improvement), Swap (Best Improvement), Replace (First Improvement), Replace (Best Improvement), and Insert (Best Improvement). These local search methods explore the solution space to find better routes by making incremental adjustments.

**Metaheuristics.py**: This module features advanced metaheuristic techniques, including Variable Neighborhood Descent (VND) and a modified Multi-Start Iterated Local Search (MS-ILS). VND systematically explores different neighborhoods to find improved solutions, while MS-ILS iterates over multiple starting solutions, applying local search techniques to escape local optima and explore the solution space more effectively.

## Getting Started

To use the algorithms provided in this repository, clone the repository and import the desired modules into your project. You can do this with the following commands:

```bash
git clone https://github.com/saamayac/C2461_TOP.git
```

```python
from Workshop.Base import *
from Workshop.Evolutionary import *
from Workshop.InitSol import *
from Workshop.LocalSearch import *
from Workshop.Metaheuristics import *
```

### Example Usage

Here is an example of how to initialize a solution and apply a local search algorithm:

```python
Top_data = Read('TOP').read_all()
file_name = 'TOP1.txt'
m, n, L, data = Top_data[file_name]
max_time_data = pd.read_excel('TimeLimit.xlsx', index_col=0,  header=None, ).to_dict()[1]

constructive(m, n, L, data, file_name).solve().dump()
random1(m, n, L, data, file_name).solve().dump()
random2(m, n, L, data, file_name).solve().dump()
swap_best(m, n, L, data, file_name).solve().dump()
swap_first(m, n, L, data, file_name).solve().dump()
replace_best(m, n, L, data, file_name).solve().dump()
replace_first(m, n, L, data, file_name).solve().dump()
insert_best(m, n, L, data, file_name).solve().dump()
vnd(m, n, L, data, file_name).solve().sol
Mod_MS_ILS(m, n, L, data, file_name, max_time=max_time_data[file_name.split('.')[0]], margin=2,
           range_init_sol=(-2,2), range_criterion=(-2,2), alpha=0.1, init_iter=1).solve().dump('test')
genetic(m, n, L, data, file_name, max_time=max_time_data[file_name.split('.')[0]], margin=2).solve().dump()
```

## Contributing

Contributions to this repository are welcome. If you have any enhancements or bug fixes, please fork the repository and submit a pull request. We appreciate your efforts to improve this project.

## License

This project is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE) file.

For any questions or issues, please open an issue on GitHub. Thank you for your interest, and happy optimizing!
