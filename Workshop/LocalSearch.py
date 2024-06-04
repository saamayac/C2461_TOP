import numpy as np
import time
from Workshop import Base, InitSol
from itertools import combinations

class localbase(Base.CodeBase):
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, algo_type: str = '', name: str = None, verbose:bool = True, **kwargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = algo_type, name = name, verbose=verbose, **kwargs)
        if 'init_sol' in kwargs.keys():
            self.sol = kwargs['init_sol']
        else:
            if 'range_init_sol' in kwargs.keys():
                self.range_init_sol = kwargs['range_init_sol']
            else:
                self.range_init_sol = (-2,2)

            if 'init_iter' in kwargs.keys():
                self.init_iter = kwargs['init_iter']
            else:
                self.init_iter = 1
            
            self.sol = InitSol.random1(self.m, self.n, self.L, self.node_info,
                                       self.name, self.range_init_sol, self.init_iter, verbose=verbose).solve().sol.copy()

        if 'time_limit' in kwargs.keys():
            self.time_limit = kwargs['time_limit']
        else:
            self.time_limit = np.inf

    def swap(self, type_: str = 'first'):
        if type_ == 'first':
            return self.swap_first()
        elif type_ == 'best': 
            return self.swap_best()
            
    def swap_best(self):
        change_flag=False
        if time.time() > self.time_limit:
            return change_flag
        stop_flag=False
        while not(stop_flag):
            stop_flag=True
            S = self.sol.copy()
            for team in S:
                sol_aux = S[team].copy()
                for i, j in combinations(range(1,self.sol[team].shape[0]-1), 2):
                    if time.time() > self.time_limit:
                        return change_flag
                    sol_aux[i], sol_aux[j] = sol_aux[j], sol_aux[i]
                    if self.total_dist_mat(sol_aux) <= self.L:
                        if  self.total_dist_mat(sol_aux) < self.total_dist_mat(self.sol[team]):
                            self.sol[team] = sol_aux.copy()
                            stop_flag=False
                            change_flag=True
        return change_flag
                            
    def swap_first(self):
        change_flag=False
        if time.time() > self.time_limit:
            return change_flag
        stop_flag=False
        while not(stop_flag):
            stop_flag=True
            for team in self.sol:
                if not(stop_flag):
                    break
                sol_aux = self.sol[team].copy()
                for i, j in combinations(range(1,self.sol[team].shape[0]-1), 2):
                    if time.time() > self.time_limit:
                        return change_flag

                    if not(stop_flag):
                        break
                    sol_aux[i], sol_aux[j] = sol_aux[j], sol_aux[i]
                    if self.total_dist_mat(sol_aux) <= self.L:
                        if  self.total_dist_mat(sol_aux) < self.total_dist_mat(self.sol[team]):
                            self.sol[team] = sol_aux.copy()
                            stop_flag=False
                            change_flag=True
        return change_flag

    def replace(self, type_: str = 'first'):
        if type_ == 'first':
            return self.replace_first()
        elif type_ == 'best':
            return self.replace_best()


    def replace_best(self):
        change_flag=False
        if time.time() > self.time_limit:
            return change_flag
        stop_flag=False
        while not(stop_flag):
            stop_flag=True
            visited = set(np.concatenate([*self.sol.values()]))
            not_visited = set(range(self.n)) - visited
            S = self.sol.copy()
            for node in not_visited:
                for team in S:
                    for index in range(1, len(S)-1):
                        if time.time() > self.time_limit:
                            return change_flag

                        sol_aux = S[team].copy()
                        sol_aux[index] = node
                        if self.total_dist_mat(sol_aux) <= self.L:
                            if (self.total_weight(self.sol[team]) < self.total_weight(sol_aux)):
                                self.sol[team] = sol_aux.copy()
                                stop_flag = False
                                change_flag=True
        return change_flag

    def replace_first(self):
        change_flag=False
        if time.time() > self.time_limit:
            return change_flag
        stop_flag=False
        while not(stop_flag):
            stop_flag=True
            visited = set(np.concatenate([*self.sol.values()]))
            not_visited = set(range(self.n)) - visited
            for node in not_visited:
                if not(stop_flag):
                    break
                for team in self.sol:
                    if not(stop_flag):
                        break
                    for index in range(1, len(self.sol[team])-1):
                        if time.time() > self.time_limit:
                            return change_flag

                        if not(stop_flag):
                            break
                        sol_aux = self.sol[team].copy()
                        sol_aux[index] = node
                        if self.total_dist_mat(sol_aux) <= self.L:
                            if (self.total_weight(self.sol[team]) < self.total_weight(sol_aux)):
                                self.sol[team] = sol_aux.copy()
                                stop_flag = False
                                change_flag=True
        return change_flag

    def insert(self):
        change_flag=False
        if time.time() > self.time_limit:
            return change_flag
        visited = set(np.concatenate([*self.sol.values()]))
        posible_nodes = set(range(self.n))
        metric = lambda S_aux, S: self.total_weight(S_aux)/(self.total_dist(S_aux)-self.total_dist(S))
        visited_size = 0
        while visited_size < len(visited):
            visited_size = len(visited)
            f_max = 0
            f_obj = 0
            index_max = None
            team_max = None
            node_max = None
            for node in (posible_nodes-visited):
                for team in self.sol.keys():
                    S = self.sol[team].copy()
                    for index in range(1, len(S)):
                        if time.time() > self.time_limit:
                            return change_flag
                        S_aux = np.insert(S, index, node)
                        if self.total_dist_mat(S_aux) <= self.L:
                            try:
                                f_obj = metric(S_aux, S)
                            except ZeroDivisionError:
                                f_obj = np.inf

                            if f_obj > f_max:
                                f_max = f_obj
                                index_max = index
                                team_max = team
                                node_max = node
            if node_max != None:
                self.sol[team_max] = np.insert(self.sol[team_max], index_max, node_max)
                visited.add(node_max)
                change_flag=True
        return change_flag


class local1(localbase):
    ALGO_TYPE = 'busqueda_local_1'
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None) -> None:
        super().__init__(m, n, L, node_info, algo_type = self.ALGO_TYPE, name = name)

    @Base.CodeBase.execution_time
    def solve_aux(self):
        self.swap('best')
        self.replace('first')
        self.insert()

class local2(localbase):
    ALGO_TYPE = 'busqueda_local_2'
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None) -> None:
        super().__init__(m, n, L, node_info, algo_type = self.ALGO_TYPE, name = name)

    @Base.CodeBase.execution_time
    def solve_aux(self):
        self.swap('first')
        self.replace('best')
        self.insert()

class local3(localbase):
    ALGO_TYPE = 'busqueda_local_3'
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None) -> None:
        super().__init__(m, n, L, node_info, algo_type = self.ALGO_TYPE, name = name)

    @Base.CodeBase.execution_time
    def solve_aux(self):
        self.swap('best')
        self.replace('best')
        self.insert()    
