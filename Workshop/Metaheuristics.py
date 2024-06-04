import numpy as np
import time
from Workshop import Base, InitSol, LocalSearch
import threading
import random
from itertools import combinations

class vnd(LocalSearch.localbase):
    ALGO_TYPE = 'VND'
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None, verbose:bool = True, **kargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = self.ALGO_TYPE, name = name, verbose = verbose, **kargs)
    
    @Base.CodeBase.execution_time
    def solve_aux(self):
        change_flag = True
        while change_flag:
            change_flag = False
            change_flag = self.swap('best')
            if change_flag: continue
            change_flag = self.replace('first')
            if change_flag: continue
            change_flag = self.insert() 

class metaheuristic1(Base.CodeBase):
    ALGO_TYPE = 'Mod_MS_ILS'
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None, **kwargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = self.ALGO_TYPE, name = name)
        self.margin = kwargs['margin']
        self.max_time = kwargs['max_time'] - self.margin
        self.range_init_sol = kwargs['range_init_sol']
        self.range_criterion = kwargs['range_criterion']
        self.init_iter = kwargs['init_iter']
        self.alpha = kwargs['alpha']
        self.vnd = vnd
        self.sol = self.init_sols_generator(verbose=True)
                
    def init_sols_generator(self, verbose:bool = False):
        if time.time() > self.time_limit:
            return self.sol.copy()
            
        return InitSol.random1(self.m, self.n, self.L, self.node_info,
                               self.name, self.range_init_sol, self.init_iter, verbose=verbose).solve().sol.copy()

    def fun_vnd(self, sol_aux, verbose:bool = False):
        if time.time() > self.time_limit:
            return sol_aux
            
        return self.vnd(self.m, self.n, self.L, self.node_info, self.name,
                        init_sol=sol_aux,
                        time_limit=self.time_limit,
                        verbose=verbose).solve().sol.copy()

    def search(self, sol, visited, metric):
        if time.time() > self.time_limit:
            return sol
        
        visited = set(np.concatenate([*sol.values()]))
        posible_nodes = set(range(self.n))
        visited_size = 0
        while visited_size < len(visited):
            visited_size = len(visited)
            f_max = 0
            f_obj = 0
            index_max = None
            team_max = None
            node_max = None
            for node in (posible_nodes-visited):
                for team in sol.keys():
                    S = sol[team].copy()
                    for index in range(1, len(S)):
                        
                        if time.time() > self.time_limit:
                            return sol
                        
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
                sol[team_max] = np.insert(sol[team_max], index_max, node_max)
                visited.add(node_max)
                change_flag=True
        return sol

    def random_pop(self, sol, k_min):
        if time.time() > self.time_limit:
            return sol
        
        if len(sol)-2 < k_min:
            k_min = len(sol)-2
        k = random.randint(k_min, len(sol)-2)
        
        idxs = np.random.choice(len(sol)-2, size=k, replace=False)+1
        mask = np.ones(len(sol), dtype=bool); mask[idxs] = False
        return sol[idxs], sol[mask]

    def perturbation(self, sol, k:int = 2):
        if time.time() > self.time_limit:
            return sol

        aux_sol = {}
        pop_values = {}
        for team in sol:
           pop_values[team], aux_sol[team] = self.random_pop(sol[team], k)
        
        visited_nodes = set(np.concatenate([*aux_sol.values()]))
        restricted_nodes = set(np.concatenate([*pop_values.values()]))
        metric = lambda S_aux, S: self.total_weight(S_aux)-self.total_weight(S)/(self.total_dist(S_aux)-self.total_dist(S))
        
        aux_sol = self.search(aux_sol, visited_nodes | restricted_nodes, metric)
        return self.search(aux_sol, visited_nodes, metric)

    def time_keeper(self, t):
        while time.time() < self.time_limit+self.margin:
            print(f'time: {time.time() - (self.time_limit - self.max_time):.2f}, outer loop: {self.counter1}, inner loop: {self.counter2}', end='\r')
            time.sleep(2)
        print(f'time: {time.time() - (self.time_limit - self.max_time):.2f}, outer loop: {self.counter1}, inner loop: {self.counter2}')
        
    @Base.CodeBase.execution_time
    def solve_aux(self):
        
        self.counter1 = 0
        self.counter2 = 0
        self.time_limit = time.time() + self.max_time
        threading.Thread(target=self.time_keeper, args=[1]).start()
        
        while time.time() < self.time_limit:
            sol_aux = self.init_sols_generator()
            sol_aux = self.fun_vnd(sol_aux)
            
            self.counter1 +=1
            if self.total_weight(np.concatenate([*self.sol.values()])) < (self.total_weight(np.concatenate([*sol_aux.values()]))
                                                                          +
                                                                          random.randint(self.range_criterion[0], self.range_criterion[1])):
                self.sol = sol_aux.copy()
            
            t2 = time.time()
            while time.time() < self.time_limit and (time.time() - t2) < self.max_time*self.alpha:
                
                sol_aux = self.perturbation(sol_aux, 2)
                sol_aux = self.fun_vnd(sol_aux)
                
                self.counter2+=1
                if self.total_weight(np.concatenate([*self.sol.values()])) < (self.total_weight(np.concatenate([*sol_aux.values()]))
                                                                              +
                                                                              random.randint(self.range_criterion[0], self.range_criterion[1])):
                    self.sol = sol_aux.copy()
